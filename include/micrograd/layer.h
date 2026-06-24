#pragma once

/**
 * @file layer.h
 * @brief Neural network layer API.
 */

#include <stdbool.h>
#include <stddef.h>

#include "micrograd/neuron.h"

/**
 * @defgroup micrograd_layer Layer API
 * @brief Neural network layer construction, evaluation, and parameters.
 * @{
 */

/**
 * @brief Layer of neurons in a neural network.
 */
typedef struct {
    size_t n_in;        /**< Number of input values consumed by each neuron. */
    size_t n_out;       /**< Number of output values produced by the layer. */
    mg_neuron* neurons; /**< Array of `n_out` neurons. */
} mg_layer;

/**
 * @brief Initialize a layer.
 *
 * @param g Graph that owns the layer's values.
 * @param l Layer to initialize.
 * @param n_in Number of input values consumed by each neuron.
 * @param n_out Number of output values produced by the layer.
 * @param non_linear Whether each neuron should use a non-linear activation
 * function.
 * @return `true` on success, `false` on allocation failure.
 */
bool mg_layer_init(mg_graph* g,
                   mg_layer* l,
                   size_t n_in,
                   size_t n_out,
                   bool non_linear);

/**
 * @brief Evaluate a layer with the given input values.
 *
 * @param g Graph that owns any intermediate and output values.
 * @param l Layer to evaluate.
 * @param x Array of `l->n_in` input values.
 * @param out Output array with at least `l->n_out` entries.
 * @return `true` on success, `false` on invalid input or allocation failure.
 */
bool mg_layer_call(mg_graph* g, mg_layer* l, mg_value** x, mg_value** out);

/**
 * @brief Free memory owned by a layer.
 *
 * @param l Layer to free.
 */
void mg_layer_free(mg_layer* l);

/**
 * @brief Get the number of trainable parameters in a layer.
 *
 * @param l Layer to inspect.
 * @return Total number of parameters across all neurons.
 */
size_t mg_layer_param_count(const mg_layer* l);

/**
 * @brief Copy a layer's parameters into a flat array.
 *
 * @param l Layer to inspect.
 * @param out Output array with at least `mg_layer_param_count(l)` entries.
 */
void mg_layer_params(const mg_layer* l, mg_value** out);

/** @} */
