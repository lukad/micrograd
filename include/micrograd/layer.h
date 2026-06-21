#pragma once

#include <stddef.h>

#include "micrograd/neuron.h"

typedef struct {
    size_t n_in;
    size_t n_out;
    mg_neuron* neurons;
} mg_layer;

bool mg_layer_init(mg_graph* g, mg_layer* l, size_t n_in, size_t n_out);
bool mg_layer_call(mg_graph* g, mg_layer* l, mg_value** x, mg_value** out);
void mg_layer_free(mg_layer* l);
size_t mg_layer_param_count(const mg_layer* l);
void mg_layer_params(const mg_layer* l, mg_value** out);
