#include <stdlib.h>

#include "micrograd/mlp.h"
#include "micrograd/value.h"

bool mg_mlp_init(mg_graph* g,
                 mg_mlp* m,
                 size_t n_in,
                 const size_t* sizes,
                 size_t n_sizes) {
    m->n_layers = n_sizes;
    m->layers = calloc(n_sizes, sizeof(*m->layers));
    if (!m->layers) {
        return false;
    }

    size_t layer_in = n_in;

    for (size_t i = 0; i < n_sizes; i++) {
        bool non_linear = i + 1 < n_sizes;
        if (!mg_layer_init(g, &m->layers[i], layer_in, sizes[i], non_linear)) {
            return false;
        }

        layer_in = sizes[i];
    }

    return true;
}

bool mg_mlp_call(mg_graph* g, mg_mlp* m, mg_value** x, mg_value** out) {
    mg_value** current = x;
    mg_value** tmp = NULL;

    for (size_t i = 0; i < m->n_layers; i++) {
        mg_layer* layer = &m->layers[i];

        mg_value** next = calloc(layer->n_out, sizeof(*next));
        if (!next) {
            free(tmp);
            return false;
        }

        if (!mg_layer_call(g, layer, current, next)) {
            free(next);
            free(tmp);
            return false;
        }

        free(tmp);
        tmp = next;
        current = tmp;
    }

    mg_layer* last = &m->layers[m->n_layers - 1];
    for (size_t i = 0; i < last->n_out; i++) {
        out[i] = current[i];
    }

    free(tmp);
    return true;
}

size_t mg_mlp_param_count(const mg_mlp* m) {
    size_t count = 0;

    for (size_t i = 0; i < m->n_layers; i++) {
        count += mg_layer_param_count(&m->layers[i]);
    }

    return count;
}

void mg_mlp_params(const mg_mlp* m, mg_value** out) {
    size_t k = 0;

    for (size_t i = 0; i < m->n_layers; i++) {
        mg_layer_params(&m->layers[i], &out[k]);
        k += mg_layer_param_count(&m->layers[i]);
    }
}

void mg_mlp_free(mg_mlp* m) {
    for (size_t i = 0; i < m->n_layers; i++) {
        mg_layer_free(&m->layers[i]);
    }

    free(m->layers);
    m->layers = NULL;
    m->n_layers = 0;
}
