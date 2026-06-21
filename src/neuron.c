#include <math.h>
#include <stdlib.h>

#include "micrograd/neuron.h"
#include "micrograd/value.h"

static float rand_uniform(float lo, float hi) {
    float t = (float)rand() / (float)RAND_MAX;
    return lo + t * (hi - lo);
}

bool mg_neuron_init(mg_graph* g, mg_neuron* n, size_t n_in) {
    n->n_in = n_in;
    n->w = calloc(n_in, sizeof(*n->w));
    if (!n->w) {
        return false;
    }

    float limit = 1.0f / sqrtf((float)n_in);

    for (size_t i = 0; i < n_in; i++) {
        n->w[i] = mg_scalar(g, rand_uniform(-limit, limit));
        if (!n->w[i]) {
            free(n->w);
            n->w = NULL;
            return false;
        }
    }

    n->b = mg_scalar(g, 0.0);
    if (!n->b) {
        free(n->w);
        n->w = NULL;
        return false;
    }

    return true;
}

void mg_neuron_free(mg_neuron* n) {
    free(n->w);
    n->w = NULL;
    n->b = NULL;
    n->n_in = 0;
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

    return mg_tanh(g, out);
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
