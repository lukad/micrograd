#pragma once

/**
 * @file neuron.h
 * @brief Fully connected neuron API.
 */

#include <stdbool.h>
#include <stddef.h>

#include "micrograd/value.h"

/**
 * @defgroup micrograd_neuron Neuron API
 * @brief Fully connected neuron construction, evaluation, and parameters.
 * @{
 */

/**
 * @brief Fully connected neuron in a neural network.
 */
typedef struct {
    size_t n_in;     /**< Number of input values. */
    bool non_linear; /**< Whether to apply the non-linear activation function. */
    mg_value** w;    /**< Array of `n_in` weight values. */
    mg_value* b;     /**< Bias value. */
} mg_neuron;

/**
 * @brief Initialize a neuron.
 *
 * @param g Graph that owns the neuron's values.
 * @param n Neuron to initialize.
 * @param nin Number of input values.
 * @param non_linear Whether to apply the non-linear activation function.
 * @return `true` on success, `false` on allocation failure.
 */
bool mg_neuron_init(mg_graph* g, mg_neuron* n, size_t nin, bool non_linear);

/**
 * @brief Free memory owned by a neuron.
 *
 * @param n Neuron to free.
 */
void mg_neuron_free(mg_neuron* n);

/**
 * @brief Evaluate a neuron with the given input values.
 *
 * @param g Graph that owns any intermediate and output values.
 * @param n Neuron to evaluate.
 * @param x Array of `n->n_in` input values.
 * @return The output value, or `NULL` on invalid input or allocation failure.
 */
mg_value* mg_neuron_call(mg_graph* g, mg_neuron* n, mg_value** x);

/**
 * @brief Get the number of trainable parameters in a neuron.
 *
 * @param n Neuron to inspect.
 * @return Number of weight values plus the bias value.
 */
size_t mg_neuron_param_count(const mg_neuron* n);

/**
 * @brief Copy a neuron's parameters into a flat array.
 *
 * @param n Neuron to inspect.
 * @param out Output array with at least `mg_neuron_param_count(n)` entries.
 */
void mg_neuron_params(const mg_neuron* n, mg_value** out);

/** @} */
