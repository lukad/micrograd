#pragma once

/**
 * @file value.h
 * @brief Scalar value and autograd graph API.
 */

#include <stdbool.h>
#include <stddef.h>

/**
 * @defgroup micrograd_value Value API
 * @brief Scalar values and automatic differentiation graph operations.
 * @{
 */

/**
 * @brief Opaque owner of a linked list of `mg_value` nodes.
 *
 * A graph frees all values it owns when passed to `mg_graph_free`.
 */
typedef struct mg_graph mg_graph;

/**
 * @brief Opaque scalar node in a computation graph.
 *
 * A value stores scalar data, its gradient, and the operation and parent values
 * that produced it.
 */
typedef struct mg_value mg_value;

/**
 * @brief Snapshot of a graph's allocation state.
 *
 * Use a checkpoint with `mg_graph_restore` to roll a graph back to an earlier
 * state.
 */
typedef struct {
    mg_value* values; /**< Head value at the checkpoint. */
    size_t len;       /**< Number of values at the checkpoint. */
} mg_graph_checkpoint;

/**
 * @brief Create a new graph.
 *
 * @return A new graph, or `NULL` if allocation fails.
 */
mg_graph* mg_graph_new(void);

/**
 * @brief Free a graph and all values it owns.
 *
 * @param g Graph to free. Passing `NULL` is allowed.
 */
void mg_graph_free(mg_graph* g);

/**
 * @brief Set all `mg_value` gradients in a graph to zero.
 *
 * @param g Graph whose values should be reset.
 */
void mg_zero_grad(mg_graph* g);

/**
 * @brief Create a snapshot of the graph's current allocation state.
 *
 * @param g Graph to snapshot.
 * @return A checkpoint that can be passed to `mg_graph_restore`.
 */
mg_graph_checkpoint mg_graph_save(const mg_graph* g);

/**
 * @brief Restore the graph to a previous checkpoint.
 *
 * @param g Graph to restore.
 * @param checkpoint Checkpoint returned by `mg_graph_save`.
 */
void mg_graph_restore(mg_graph* g, mg_graph_checkpoint checkpoint);

/**
 * @brief Get the scalar data of a value.
 *
 * @param v Value to inspect.
 * @return The scalar data stored in `v`.
 */
float mg_data(const mg_value* v);

/**
 * @brief Get the gradient of a value.
 *
 * @param v Value to inspect.
 * @return The gradient stored in `v`.
 */
float mg_grad(const mg_value* v);

/**
 * @brief Set the scalar data of a value.
 *
 * @param v Value to update.
 * @param data New scalar data.
 */
void mg_set_data(mg_value* v, float data);

/**
 * @brief Set the gradient of a value.
 *
 * @param v Value to update.
 * @param grad New gradient.
 */
void mg_set_grad(mg_value* v, float grad);

/**
 * @brief Create a new scalar value.
 *
 * @param g Graph that owns the new value.
 * @param data Initial scalar data.
 * @return The new value, or `NULL` if allocation fails.
 */
mg_value* mg_scalar(mg_graph* g, float data);

/**
 * @brief Add two values.
 *
 * @param g Graph that owns the result.
 * @param a Left operand.
 * @param b Right operand.
 * @return The sum, or `NULL` if allocation fails.
 */
mg_value* mg_add(mg_graph* g, mg_value* a, mg_value* b);

/**
 * @brief Subtract one value from another.
 *
 * @param g Graph that owns the result.
 * @param a Minuend.
 * @param b Subtrahend.
 * @return The difference, or `NULL` if allocation fails.
 */
mg_value* mg_sub(mg_graph* g, mg_value* a, mg_value* b);

/**
 * @brief Multiply two values.
 *
 * @param g Graph that owns the result.
 * @param a Left operand.
 * @param b Right operand.
 * @return The product, or `NULL` if allocation fails.
 */
mg_value* mg_mul(mg_graph* g, mg_value* a, mg_value* b);

/**
 * @brief Divide one value by another.
 *
 * @param g Graph that owns the result.
 * @param a Dividend.
 * @param b Divisor.
 * @return The quotient, or `NULL` if allocation fails.
 */
mg_value* mg_div(mg_graph* g, mg_value* a, mg_value* b);

/**
 * @brief Raise one value to the power of another.
 *
 * @param g Graph that owns the result.
 * @param a Base.
 * @param b Exponent.
 * @return The power result, or `NULL` if allocation fails.
 */
mg_value* mg_pow(mg_graph* g, mg_value* a, mg_value* b);

/**
 * @brief Negate a value.
 *
 * @param g Graph that owns the result.
 * @param a Value to negate.
 * @return The negated value, or `NULL` if allocation fails.
 */
mg_value* mg_neg(mg_graph* g, mg_value* a);

/**
 * @brief Square a value.
 *
 * @param g Graph that owns the result.
 * @param a Value to square.
 * @return The squared value, or `NULL` if allocation fails.
 */
mg_value* mg_square(mg_graph* g, mg_value* a);

/**
 * @brief Apply the ReLU activation function to a value.
 *
 * @param g Graph that owns the result.
 * @param a Input value.
 * @return The activated value, or `NULL` if allocation fails.
 */
mg_value* mg_relu(mg_graph* g, mg_value* a);

/**
 * @brief Apply the tanh activation function to a value.
 *
 * @param g Graph that owns the result.
 * @param a Input value.
 * @return The activated value, or `NULL` if allocation fails.
 */
mg_value* mg_tanh(mg_graph* g, mg_value* a);

/**
 * @brief Apply the exponential function to a value.
 *
 * @param g Graph that owns the result.
 * @param a Input value.
 * @return The exponential value, or `NULL` if allocation fails.
 */
mg_value* mg_exp(mg_graph* g, mg_value* a);

/**
 * @brief Compute gradients for all values that contribute to an output value.
 *
 * @param g Graph containing the values.
 * @param out Output value to differentiate from.
 * @return `true` on success, `false` if allocation fails.
 */
bool mg_backward(mg_graph* g, mg_value* out);

/** @} */
