#include <math.h>
#include <stdbool.h>
#include <stdio.h>
#include <stdlib.h>

#include "micrograd/mlp.h"
#include "micrograd/value.h"

static bool closef(float a, float b) {
    return fabsf(a - b) < 1e-5f;
}

static void expect_close(const char* name, float actual, float expected) {
    if (closef(actual, expected)) {
        return;
    }

    fprintf(stderr, "%s: expected %f, got %f\n", name, expected, actual);
    exit(EXIT_FAILURE);
}

static void expect_true(const char* name, bool ok) {
    if (ok) {
        return;
    }

    fprintf(stderr, "%s: expected true\n", name);
    exit(EXIT_FAILURE);
}

static void test_add(void) {
    mg_graph* g = mg_graph_new();
    mg_value* a = mg_scalar(g, 2.0f);
    mg_value* b = mg_scalar(g, 3.0f);
    mg_value* c = mg_add(g, a, b);

    if (!mg_backward(g, c)) {
        fprintf(stderr, "mg_backward failed\n");
        exit(EXIT_FAILURE);
    }

    expect_close("add data", mg_data(c), 5.0f);
    expect_close("add da", mg_grad(a), 1.0f);
    expect_close("add db", mg_grad(b), 1.0f);

    mg_graph_free(g);
}

static void test_sub(void) {
    mg_graph* g = mg_graph_new();
    mg_value* a = mg_scalar(g, 5.0f);
    mg_value* b = mg_scalar(g, 3.0f);
    mg_value* c = mg_sub(g, a, b);

    if (!mg_backward(g, c)) {
        fprintf(stderr, "mg_backward failed\n");
        exit(EXIT_FAILURE);
    }

    expect_close("sub data", mg_data(c), 2.0f);
    expect_close("sub da", mg_grad(a), 1.0f);
    expect_close("sub db", mg_grad(b), -1.0f);

    mg_graph_free(g);
}

static void test_mul(void) {
    mg_graph* g = mg_graph_new();
    mg_value* a = mg_scalar(g, 2.0f);
    mg_value* b = mg_scalar(g, 3.0f);
    mg_value* c = mg_mul(g, a, b);

    if (!mg_backward(g, c)) {
        fprintf(stderr, "mg_backward failed\n");
        exit(EXIT_FAILURE);
    }

    expect_close("mul data", mg_data(c), 6.0f);
    expect_close("mul da", mg_grad(a), 3.0f);
    expect_close("mul db", mg_grad(b), 2.0f);

    mg_graph_free(g);
}

static void test_mul_same_input(void) {
    mg_graph* g = mg_graph_new();
    mg_value* a = mg_scalar(g, 3.0f);
    mg_value* c = mg_mul(g, a, a);

    if (!mg_backward(g, c)) {
        fprintf(stderr, "mg_backward failed\n");
        exit(EXIT_FAILURE);
    }

    expect_close("square data", mg_data(c), 9.0f);
    expect_close("square da", mg_grad(a), 6.0f);

    mg_graph_free(g);
}

static void test_pow(void) {
    mg_graph* g = mg_graph_new();
    mg_value* base = mg_scalar(g, 4.0f);
    mg_value* exponent = mg_scalar(g, 3.0f);
    mg_value* out = mg_pow(g, base, exponent);

    if (!mg_backward(g, out)) {
        fprintf(stderr, "mg_backward failed\n");
        exit(EXIT_FAILURE);
    }

    expect_close("pow data", mg_data(out), 64.0f);
    expect_close("pow base grad", mg_grad(base), 48.0f);
    expect_close("pow exponent grad", mg_grad(exponent), 64.0f * logf(4.0f));

    mg_graph_free(g);
}

static void test_div(void) {
    mg_graph* g = mg_graph_new();
    mg_value* a = mg_scalar(g, 6.0f);
    mg_value* b = mg_scalar(g, 3.0f);
    mg_value* c = mg_div(g, a, b);

    if (!mg_backward(g, c)) {
        fprintf(stderr, "mg_backward failed\n");
        exit(EXIT_FAILURE);
    }

    expect_close("div data", mg_data(c), 2.0f);
    expect_close("div da", mg_grad(a), 1.0f / 3.0f);
    expect_close("div db", mg_grad(b), -2.0f / 3.0f);

    mg_graph_free(g);
}

static void test_relu(void) {
    mg_graph* g = mg_graph_new();
    mg_value* a = mg_scalar(g, -2.0f);
    mg_value* b = mg_scalar(g, 3.0f);
    mg_value* c = mg_relu(g, a);
    mg_value* d = mg_relu(g, b);

    if (!mg_backward(g, c)) {
        fprintf(stderr, "mg_backward failed\n");
        exit(EXIT_FAILURE);
    }

    expect_close("relu negative data", mg_data(c), 0.0f);
    expect_close("relu negative grad", mg_grad(a), 0.0f);

    if (!mg_backward(g, d)) {
        fprintf(stderr, "mg_backward failed\n");
        exit(EXIT_FAILURE);
    }

    expect_close("relu positive data", mg_data(d), 3.0f);
    expect_close("relu positive grad", mg_grad(b), 1.0f);

    mg_graph_free(g);
}

static void test_tanh(void) {
    mg_graph* g = mg_graph_new();
    mg_value* a = mg_scalar(g, 0.5f);
    mg_value* b = mg_tanh(g, a);

    if (!mg_backward(g, b)) {
        fprintf(stderr, "mg_backward failed\n");
        exit(EXIT_FAILURE);
    }

    float expected = tanhf(0.5f);

    expect_close("tanh data", mg_data(b), expected);
    expect_close("tanh grad", mg_grad(a), 1.0f - expected * expected);

    mg_graph_free(g);
}

static void test_exp(void) {
    mg_graph* g = mg_graph_new();
    mg_value* a = mg_scalar(g, 0.5f);
    mg_value* b = mg_exp(g, a);

    if (!mg_backward(g, b)) {
        fprintf(stderr, "mg_backward failed\n");
        exit(EXIT_FAILURE);
    }

    float expected = expf(0.5f);

    expect_close("exp data", mg_data(b), expected);
    expect_close("exp grad", mg_grad(a), expected);

    mg_graph_free(g);
}

static void test_graph_checkpoint_restore(void) {
    mg_graph* g = mg_graph_new();
    mg_value* p = mg_scalar(g, 1.0f);
    mg_graph_checkpoint checkpoint = mg_graph_save(g);

    mg_value* shifted = mg_add(g, p, mg_scalar(g, 2.0f));
    mg_value* loss = mg_square(g, shifted);

    if (!mg_backward(g, loss)) {
        fprintf(stderr, "mg_backward failed\n");
        exit(EXIT_FAILURE);
    }

    mg_set_data(p, mg_data(p) - 0.1f * mg_grad(p));
    expect_close("checkpoint updated param", mg_data(p), 0.4f);

    mg_graph_restore(g, checkpoint);

    mg_value* out = mg_mul(g, p, mg_scalar(g, 3.0f));

    if (!mg_backward(g, out)) {
        fprintf(stderr, "mg_backward failed\n");
        exit(EXIT_FAILURE);
    }

    expect_close("checkpoint reused param data", mg_data(out), 1.2f);
    expect_close("checkpoint reused param grad", mg_grad(p), 3.0f);

    mg_graph_free(g);
}

static void test_training_checkpoint_reuse(void) {
    mg_graph* g = mg_graph_new();
    mg_mlp m = {0};
    size_t sizes[] = {1};

    if (!mg_mlp_init(g, &m, 1, sizes, 1)) {
        fprintf(stderr, "mg_mlp_init failed\n");
        exit(EXIT_FAILURE);
    }

    size_t n_params = mg_mlp_param_count(&m);
    mg_value* params[2];
    mg_mlp_params(&m, params);

    for (size_t i = 0; i < n_params; i++) {
        mg_set_data(params[i], 0.0f);
    }

    mg_graph_checkpoint checkpoint = mg_graph_save(g);

    mg_value* x0[] = {mg_scalar(g, 1.0f)};
    mg_value* out0[1];

    if (!mg_mlp_call(g, &m, x0, out0)) {
        fprintf(stderr, "mg_mlp_call failed\n");
        exit(EXIT_FAILURE);
    }

    mg_value* diff0 = mg_sub(g, out0[0], mg_scalar(g, 1.0f));
    mg_value* loss0 = mg_square(g, diff0);

    if (!mg_backward(g, loss0)) {
        fprintf(stderr, "mg_backward failed\n");
        exit(EXIT_FAILURE);
    }

    float initial_loss = mg_data(loss0);

    for (size_t i = 0; i < n_params; i++) {
        mg_set_data(params[i], mg_data(params[i]) - 0.1f * mg_grad(params[i]));
    }

    expect_true("checkpoint param changed",
                fabsf(mg_data(params[0])) > 1e-5f ||
                    fabsf(mg_data(params[1])) > 1e-5f);

    mg_graph_restore(g, checkpoint);

    mg_value* x1[] = {mg_scalar(g, 1.0f)};
    mg_value* out1[1];

    if (!mg_mlp_call(g, &m, x1, out1)) {
        fprintf(stderr, "mg_mlp_call failed\n");
        exit(EXIT_FAILURE);
    }

    mg_value* diff1 = mg_sub(g, out1[0], mg_scalar(g, 1.0f));
    mg_value* loss1 = mg_square(g, diff1);

    if (!mg_backward(g, loss1)) {
        fprintf(stderr, "mg_backward failed\n");
        exit(EXIT_FAILURE);
    }

    expect_true("checkpoint fresh loss improved", mg_data(loss1) < initial_loss);

    mg_mlp_free(&m);
    mg_graph_free(g);
}

static void test_neuron(void) {
    mg_graph* g = mg_graph_new();
    mg_neuron n = {0};

    if (!mg_neuron_init(g, &n, 2)) {
        fprintf(stderr, "mg_neuron_init failed\n");
        exit(EXIT_FAILURE);
    }

    mg_value* params[3];
    mg_neuron_params(&n, params);
    mg_set_data(params[0], 0.5f);
    mg_set_data(params[1], -1.0f);
    mg_set_data(params[2], 0.25f);

    mg_value* x[] = {
        mg_scalar(g, 2.0f),
        mg_scalar(g, 3.0f),
    };
    mg_value* out = mg_neuron_call(g, &n, x);
    float preact = 0.5f * 2.0f + -1.0f * 3.0f + 0.25f;
    float local_grad = 1.0f - tanhf(preact) * tanhf(preact);

    if (!mg_backward(g, out)) {
        fprintf(stderr, "mg_backward failed\n");
        exit(EXIT_FAILURE);
    }

    expect_close("neuron param count", mg_neuron_param_count(&n), 3.0f);
    expect_close("neuron data", mg_data(out), tanhf(preact));
    expect_close("neuron w0 grad", mg_grad(params[0]), 2.0f * local_grad);
    expect_close("neuron w1 grad", mg_grad(params[1]), 3.0f * local_grad);
    expect_close("neuron bias grad", mg_grad(params[2]), local_grad);
    expect_close("neuron x0 grad", mg_grad(x[0]), 0.5f * local_grad);
    expect_close("neuron x1 grad", mg_grad(x[1]), -1.0f * local_grad);

    mg_neuron_free(&n);
    mg_graph_free(g);
}

static void test_layer(void) {
    mg_graph* g = mg_graph_new();
    mg_layer l = {0};

    if (!mg_layer_init(g, &l, 2, 3)) {
        fprintf(stderr, "mg_layer_init failed\n");
        exit(EXIT_FAILURE);
    }

    mg_value* params[9];
    mg_layer_params(&l, params);
    float values[] = {
        1.0f, 0.0f, 0.0f,
        0.0f, 1.0f, 0.0f,
        1.0f, 1.0f, -1.0f,
    };

    for (size_t i = 0; i < 9; i++) {
        mg_set_data(params[i], values[i]);
    }

    mg_value* x[] = {
        mg_scalar(g, 2.0f),
        mg_scalar(g, 3.0f),
    };
    mg_value* out[3];

    if (!mg_layer_call(g, &l, x, out)) {
        fprintf(stderr, "mg_layer_call failed\n");
        exit(EXIT_FAILURE);
    }

    expect_close("layer param count", mg_layer_param_count(&l), 9.0f);
    expect_close("layer out0", mg_data(out[0]), tanhf(2.0f));
    expect_close("layer out1", mg_data(out[1]), tanhf(3.0f));
    expect_close("layer out2", mg_data(out[2]), tanhf(4.0f));

    mg_layer_free(&l);
    mg_graph_free(g);
}

static void test_mlp(void) {
    mg_graph* g = mg_graph_new();
    mg_mlp m = {0};
    size_t sizes[] = {2, 1};

    if (!mg_mlp_init(g, &m, 2, sizes, 2)) {
        fprintf(stderr, "mg_mlp_init failed\n");
        exit(EXIT_FAILURE);
    }

    mg_value* params[9];
    mg_mlp_params(&m, params);
    float values[] = {
        1.0f, 0.0f, 0.0f,
        0.0f, 1.0f, 0.0f,
        1.0f, 1.0f, 0.0f,
    };

    for (size_t i = 0; i < 9; i++) {
        mg_set_data(params[i], values[i]);
    }

    mg_value* x[] = {
        mg_scalar(g, 0.25f),
        mg_scalar(g, -0.5f),
    };
    mg_value* out[1];

    if (!mg_mlp_call(g, &m, x, out)) {
        fprintf(stderr, "mg_mlp_call failed\n");
        exit(EXIT_FAILURE);
    }

    float hidden0 = tanhf(0.25f);
    float hidden1 = tanhf(-0.5f);

    expect_close("mlp param count", mg_mlp_param_count(&m), 9.0f);
    expect_close("mlp out", mg_data(out[0]), tanhf(hidden0 + hidden1));

    mg_mlp_free(&m);
    mg_graph_free(g);
}

int main(void) {
    test_add();
    test_sub();
    test_mul();
    test_mul_same_input();
    test_pow();
    test_div();
    test_relu();
    test_tanh();
    test_exp();
    test_graph_checkpoint_restore();
    test_training_checkpoint_reuse();
    test_neuron();
    test_layer();
    test_mlp();

    return EXIT_SUCCESS;
}
