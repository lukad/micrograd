#pragma once

/**
 * @file mlp.h
 * @brief Multi-layer perceptron API.
 */

#include <stddef.h>

#include "micrograd/layer.h"

/**
 * @defgroup micrograd_mlp MLP API
 * @brief Multi-layer perceptron construction, evaluation, and parameters.
 * @{
 */

/**
 * @brief Multi-layer perceptron model.
 */
typedef struct {
    size_t n_layers;  /**< Number of layers in the model. */
    mg_layer* layers; /**< Array of `n_layers` layers. */
} mg_mlp;

/**
 * @brief Initialize an MLP model.
 *
 * @param g Graph that owns the model's values.
 * @param m Model to initialize.
 * @param n_in Number of input features.
 * @param sizes Array containing each layer's output size.
 * @param n_sizes Number of entries in `sizes`.
 * @return `true` on success, `false` on allocation failure.
 */
bool mg_mlp_init(mg_graph* g,
                 mg_mlp* m,
                 size_t n_in,
                 const size_t* sizes,
                 size_t n_sizes);

/**
 * @brief Evaluate an MLP model.
 *
 * Calls each layer in the model sequentially.
 *
 * @param g Graph that owns any intermediate and output values.
 * @param m Model to evaluate.
 * @param x Array of input values.
 * @param out Output array with enough entries for the final layer.
 * @return `true` on success, `false` on invalid input or allocation failure.
 */
bool mg_mlp_call(mg_graph* g, mg_mlp* m, mg_value** x, mg_value** out);

/**
 * @brief Free memory owned by an MLP model.
 *
 * @param m Model to free.
 */
void mg_mlp_free(mg_mlp* m);

/**
 * @brief Get the number of trainable parameters in an MLP model.
 *
 * @param m Model to inspect.
 * @return Total number of parameters across all layers.
 */
size_t mg_mlp_param_count(const mg_mlp* m);

/**
 * @brief Copy an MLP model's parameters into a flat array.
 *
 * @param m Model to inspect.
 * @param out Output array with at least `mg_mlp_param_count(m)` entries.
 */
void mg_mlp_params(const mg_mlp* m, mg_value** out);

/** @} */
