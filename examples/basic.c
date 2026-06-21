#include <stdio.h>
#include <stdlib.h>
#include <time.h>

#include "micrograd/mlp.h"
#include "micrograd/value.h"

int main([[maybe_unused]] int argc, [[maybe_unused]] char** argv) {
    srand((unsigned)time(NULL));

    mg_graph* g = mg_graph_new();

    size_t sizes[] = {4, 4, 1};
    mg_mlp n = {0};
    if (!mg_mlp_init(g, &n, 3, sizes, 3)) {
        puts("mg_mlp_init failed");
        return EXIT_FAILURE;
    }

    float xs[][3] = {
        {2.0, 3.0, -1.0}, {3.0, -1.0, 0.5}, {0.5, 1.0, 1.0}, {1.0, 1.0, -1.0}};

    float ys[] = {1.0, -1.0, -1.0, 1.0};

    size_t n_params = mg_mlp_param_count(&n);
    mg_value** params = malloc(n_params * sizeof(*params));
    if (!params) {
        puts("could not allocate params");
        return EXIT_FAILURE;
    }

    mg_mlp_params(&n, params);
    mg_graph_checkpoint params_checkpoint = mg_graph_save(g);

    for (size_t step = 0; step < 10; step++) {
        mg_value* loss = mg_scalar(g, 0.0f);

        for (size_t i = 0; i < 4; i++) {
            mg_value* x[] = {
                mg_scalar(g, xs[i][0]),
                mg_scalar(g, xs[i][1]),
                mg_scalar(g, xs[i][2]),
            };

            mg_value* out[1];
            if (!mg_mlp_call(g, &n, x, out)) {
                puts("mg_mlp_call failed");
                free(params);
                return EXIT_FAILURE;
            }

            mg_value* ygt = mg_scalar(g, ys[i]);
            mg_value* diff = mg_sub(g, out[0], ygt);
            mg_value* sq = mg_square(g, diff);

            loss = mg_add(g, loss, sq);
        }

        if (!mg_backward(g, loss)) {
            puts("mg_backward failed");
            free(params);
            return EXIT_FAILURE;
        }

        for (size_t i = 0; i < n_params; i++) {
            float updated = mg_data(params[i]) - 0.01f * mg_grad(params[i]);
            mg_set_data(params[i], updated);
        }

        printf("step %zu loss %f\n", step, mg_data(loss));

        mg_graph_restore(g, params_checkpoint);
    }

    mg_mlp_free(&n);
    mg_graph_free(g);

    return EXIT_SUCCESS;
}
