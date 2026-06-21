#include <float.h>
#include <math.h>
#include <stddef.h>
#include <stdlib.h>

#include "micrograd/value.h"

typedef enum {
    MG_OP_NONE,
    MG_OP_ADD,
    MG_OP_MUL,
    MG_OP_POW,
    MG_OP_RELU,
    MG_OP_TANH,
    MG_OP_EXP,
} mg_op;

struct mg_graph {
    mg_value* values;
    size_t len;
};

struct mg_value {
    float data;
    float grad;

    mg_value* left;
    mg_value* right;
    mg_op op;

    bool visited;
    mg_value* next;
};

mg_graph* mg_graph_new(void) {
    mg_graph* g = calloc(1, sizeof(*g));
    return g;
}

void mg_graph_free(mg_graph* g) {
    if (!g) {
        return;
    }

    mg_value* v = g->values;
    while (v) {
        mg_value* next = v->next;
        free(v);
        v = next;
    }

    free(g);
}

void mg_zero_grad(mg_graph* g) {
    mg_value* v = g->values;
    while (v) {
        mg_set_grad(v, 0.0);
        v = v->next;
    }
}

mg_graph_checkpoint mg_graph_save(const mg_graph* g) {
    return (mg_graph_checkpoint){
        .values = g->values,
        .len = g->len,
    };
}

void mg_graph_restore(mg_graph* g, mg_graph_checkpoint checkpoint) {
    mg_value* v = g->values;
    while (v != checkpoint.values) {
        mg_value* next = v->next;
        free(v);
        v = next;
    }
    g->values = checkpoint.values;
    g->len = checkpoint.len;
}

static mg_value* mg_value_alloc(mg_graph* g, double data) {
    mg_value* v = calloc(1, sizeof(*v));
    if (!v) {
        return NULL;
    }

    v->data = data;
    v->next = g->values;
    g->values = v;
    g->len++;

    return v;
}

float mg_data(const mg_value* v) {
    return v->data;
}

float mg_grad(const mg_value* v) {
    return v->grad;
}

void mg_set_data(mg_value* v, float data) {
    v->data = data;
}

void mg_set_grad(mg_value* v, float grad) {
    v->grad = grad;
}

mg_value* mg_scalar(mg_graph* g, float data) {
    return mg_value_alloc(g, data);
}

mg_value* mg_add(mg_graph* g, mg_value* a, mg_value* b) {
    mg_value* out = mg_value_alloc(g, a->data + b->data);
    if (!out) {
        return NULL;
    }

    out->left = a;
    out->right = b;
    out->op = MG_OP_ADD;

    return out;
}

mg_value* mg_sub(mg_graph* g, mg_value* a, mg_value* b) {
    return mg_add(g, a, mg_neg(g, b));
}

mg_value* mg_mul(mg_graph* g, mg_value* a, mg_value* b) {
    mg_value* out = mg_value_alloc(g, a->data * b->data);
    if (!out) {
        return NULL;
    }

    out->left = a;
    out->right = b;
    out->op = MG_OP_MUL;

    return out;
}

mg_value* mg_div(mg_graph* g, mg_value* a, mg_value* b) {
    mg_value* inv = mg_pow(g, b, mg_scalar(g, -1.0));
    if (!inv) {
        return NULL;
    }

    return mg_mul(g, a, inv);
}

mg_value* mg_pow(mg_graph* g, mg_value* a, mg_value* b) {
    mg_value* out = mg_value_alloc(g, powf(a->data, b->data));
    if (!out) {
        return NULL;
    }

    out->left = a;
    out->right = b;
    out->op = MG_OP_POW;

    return out;
}

mg_value* mg_neg(mg_graph* g, mg_value* a) {
    return mg_mul(g, a, mg_scalar(g, -1.0));
}

mg_value* mg_square(mg_graph* g, mg_value* a) {
    return mg_mul(g, a, a);
}

mg_value* mg_relu(mg_graph* g, mg_value* a) {
    mg_value* out = mg_value_alloc(g, fmaxf(a->data, 0.0));
    if (!out) {
        return NULL;
    }

    out->left = a;
    out->right = NULL;
    out->op = MG_OP_RELU;

    return out;
}

mg_value* mg_tanh(mg_graph* g, mg_value* a) {
    mg_value* out = mg_value_alloc(g, tanhf(a->data));
    if (!out) {
        return NULL;
    }

    out->left = a;
    out->right = NULL;
    out->op = MG_OP_TANH;

    return out;
}

mg_value* mg_exp(mg_graph* g, mg_value* a) {
    mg_value* out = mg_value_alloc(g, expf(a->data));
    if (!out) {
        return NULL;
    }

    out->left = a;
    out->right = NULL;
    out->op = MG_OP_EXP;

    return out;
}

static void topo_visit(mg_value* v, mg_value** topo, size_t* len) {
    if (!v || v->visited) {
        return;
    }

    v->visited = true;

    topo_visit(v->left, topo, len);
    topo_visit(v->right, topo, len);

    topo[(*len)++] = v;
}

bool mg_backward(mg_graph* g, mg_value* out) {
    for (mg_value* v = g->values; v; v = v->next) {
        v->grad = 0.0;
        v->visited = false;
    }

    mg_value** topo = malloc(g->len * sizeof(*topo));
    if (!topo) {
        return false;
    }

    size_t len = 0;
    topo_visit(out, topo, &len);

    out->grad = 1.0;

    for (size_t i = len; i > 0; i--) {
        mg_value* v = topo[i - 1];
        switch (v->op) {
            case MG_OP_NONE:
                break;
            case MG_OP_ADD:
                v->left->grad += v->grad;
                v->right->grad += v->grad;
                break;
            case MG_OP_MUL:
                v->left->grad += v->right->data * v->grad;
                v->right->grad += v->left->data * v->grad;
                break;
            case MG_OP_POW:
                v->left->grad += v->right->data *
                                 powf(v->left->data, v->right->data - 1.0f) *
                                 v->grad;
                v->right->grad += v->data * logf(v->left->data) * v->grad;
                break;
            case MG_OP_RELU:
                v->left->grad += (v->data > 0.0) * v->grad;
                break;
            case MG_OP_TANH:
                v->left->grad += (1.0f - v->data * v->data) * v->grad;
                break;
            case MG_OP_EXP:
                v->left->grad += v->data * v->grad;
                break;
        }
    }

    free(topo);

    return true;
}
