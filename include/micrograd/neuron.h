#pragma once

#include <stdbool.h>
#include <stddef.h>

#include "micrograd/value.h"

typedef struct {
    size_t n_in;
    bool non_linear;
    mg_value** w;
    mg_value* b;
} mg_neuron;

bool mg_neuron_init(mg_graph* g, mg_neuron* n, size_t nin, bool non_linear);
void mg_neuron_free(mg_neuron* n);
mg_value* mg_neuron_call(mg_graph* g, mg_neuron* n, mg_value** x);
size_t mg_neuron_param_count(const mg_neuron* n);
void mg_neuron_params(const mg_neuron* n, mg_value** out);
