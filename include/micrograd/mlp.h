#pragma once

#include <stddef.h>

#include "micrograd/layer.h"

typedef struct {
    size_t n_layers;
    mg_layer* layers;
} mg_mlp;

bool mg_mlp_init(mg_graph* g,
                 mg_mlp* m,
                 size_t n_in,
                 const size_t* sizes,
                 size_t n_sizes);
bool mg_mlp_call(mg_graph* g, mg_mlp* m, mg_value** x, mg_value** out);
void mg_mlp_free(mg_mlp* m);
size_t mg_mlp_param_count(const mg_mlp* m);
void mg_mlp_params(const mg_mlp* m, mg_value** out);
