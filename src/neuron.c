#include <math.h>
#include <stdlib.h>

#include "micrograd/neuron.h"
#include "micrograd/value.h"

static float rand_uniform(float lo, float hi) {
    float t = (float)rand() / (float)RAND_MAX;
    return lo + t * (hi - lo);
}

bool mg_neuron_init(mg_graph* g, mg_neuron* n, size_t n_in, bool non_linear) {
    mg_graph_checkpoint checkpoint = mg_graph_save(g);
    *n = (mg_neuron){0};

    n->w = calloc(n_in, sizeof(*n->w));
    if (!n->w) {
        goto fail;
    }

    n->n_in = n_in;
    n->non_linear = non_linear;

    float limit = 1.0f / sqrtf((float)n_in);

    for (size_t i = 0; i < n_in; i++) {
        n->w[i] = mg_scalar(g, rand_uniform(-limit, limit));
        if (!n->w[i]) {
            goto fail;
        }
    }

    n->b = mg_scalar(g, 0.0);
    if (!n->b) {
        goto fail;
    }

    return true;

fail:
    mg_neuron_free(n);
    mg_graph_restore(g, checkpoint);
    return false;
}

void mg_neuron_free(mg_neuron* n) {
    free(n->w);
    n->w = NULL;
    n->b = NULL;
    n->n_in = 0;
    n->non_linear = false;
}

mg_value* mg_neuron_call(mg_graph* g, mg_neuron* n, mg_value** x) {
    if (!g || !n || !n->b || !n->w) {
        return NULL;
    }

    if (n->n_in > 0 && !x) {
        return NULL;
    }

    mg_value* out = n->b;

    for (size_t i = 0; i < n->n_in; i++) {
        if (!n->w[i] || !x[i]) {
            return NULL;
        }

        mg_value* wx = mg_mul(g, n->w[i], x[i]);
        if (!wx) {
            return NULL;
        }

        out = mg_add(g, out, wx);
        if (!out) {
            return NULL;
        }
    }

    if (n->non_linear) {
        return mg_tanh(g, out);
    }

    return out;
}

size_t mg_neuron_param_count(const mg_neuron* n) {
    return n->n_in + 1;
}

void mg_neuron_params(const mg_neuron* n, mg_value** out) {
    for (size_t i = 0; i < n->n_in; i++) {
        out[i] = n->w[i];
    }

    out[n->n_in] = n->b;
}
