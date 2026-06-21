# micrograd

This is a small C implementation of Andrej Karpathy's micrograd, based on his [spelled-out intro to neural networks and backpropagation](https://www.youtube.com/watch?v=VMj-3S1tku0) video.

It is a learning project, not a production autodiff library.

## Build

```sh
cmake --preset debug
cmake --build --preset debug
ctest --preset debug
./build/debug/micrograd_basic
```

## Example

```c
mg_graph* g = mg_graph_new();

mg_value* a = mg_scalar(g, 2.0f);
mg_value* b = mg_scalar(g, 3.0f);
mg_value* c = mg_mul(g, a, b);

mg_backward(g, c);

printf("%f %f\n", mg_data(c), mg_grad(a));

mg_graph_free(g);
```

See [`examples/basic.c`](examples/basic.c) for a small MLP training example.

## Notes

`mg_graph` owns all `mg_value`s. Neurons, layers, and MLPs own their arrays, but not scalar values; free them before freeing the graph.

`mg_backward` resets gradients before backprop. Hidden MLP layers use `tanh`; the output layer is linear. `mg_pow` supports exponent gradients for positive bases.
