#pragma once

#include <stdbool.h>
#include <stddef.h>

typedef struct mg_graph mg_graph;
typedef struct mg_value mg_value;

typedef struct {
    mg_value* values;
    size_t len;
} mg_graph_checkpoint;

mg_graph* mg_graph_new(void);
void mg_graph_free(mg_graph* g);
void mg_zero_grad(mg_graph* g);

mg_graph_checkpoint mg_graph_save(const mg_graph* g);
void mg_graph_restore(mg_graph* g, mg_graph_checkpoint checkpoint);

float mg_data(const mg_value* v);
float mg_grad(const mg_value* v);

void mg_set_data(mg_value* v, float data);
void mg_set_grad(mg_value* v, float grad);

mg_value* mg_scalar(mg_graph* g, float data);
mg_value* mg_add(mg_graph* g, mg_value* a, mg_value* b);
mg_value* mg_sub(mg_graph* g, mg_value* a, mg_value* b);
mg_value* mg_mul(mg_graph* g, mg_value* a, mg_value* b);
mg_value* mg_div(mg_graph* g, mg_value* a, mg_value* b);
mg_value* mg_pow(mg_graph* g, mg_value* a, mg_value* b);
mg_value* mg_neg(mg_graph* g, mg_value* a);
mg_value* mg_square(mg_graph* g, mg_value* a);
mg_value* mg_relu(mg_graph* g, mg_value* a);
mg_value* mg_tanh(mg_graph* g, mg_value* a);
mg_value* mg_exp(mg_graph* g, mg_value* a);

bool mg_backward(mg_graph* g, mg_value* out);
