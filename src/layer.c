#include <stdlib.h>

#include "micrograd/layer.h"
#include "micrograd/value.h"

bool mg_layer_init(mg_graph* g, mg_layer* l, size_t n_in, size_t n_out) {
    l->n_in = n_in;
    l->n_out = n_out;
    l->neurons = calloc(n_out, sizeof(*l->neurons));
    if (!l->neurons) {
        return false;
    }

    for (size_t i = 0; i < n_out; i++) {
        if (!mg_neuron_init(g, &l->neurons[i], n_in)) {
            return false;
        }
    }

    return true;
}

bool mg_layer_call(mg_graph* g, mg_layer* l, mg_value** x, mg_value** out) {
    for (size_t i = 0; i < l->n_out; i++) {
        out[i] = mg_neuron_call(g, &l->neurons[i], x);
        if (!out[i]) {
            return false;
        }
    }

    return true;
}

void mg_layer_free(mg_layer* l) {
    for (size_t i = 0; i < l->n_out; i++) {
        mg_neuron_free(&l->neurons[i]);
    }

    free(l->neurons);
    l->neurons = NULL;
    l->n_in = 0;
    l->n_out = 0;
}

size_t mg_layer_param_count(const mg_layer* l) {
    return l->n_out * (l->n_in + 1);
}

void mg_layer_params(const mg_layer* l, mg_value** out) {
    size_t k = 0;

    for (size_t i = 0; i < l->n_out; i++) {
        mg_neuron_params(&l->neurons[i], &out[k]);
        k += mg_neuron_param_count(&l->neurons[i]);
    }
}
