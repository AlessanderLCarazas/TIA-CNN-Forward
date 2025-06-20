#ifndef DENSE_LAYER_H
#define DENSE_LAYER_H

#include <vector>

class DenseLayer {
public:
    DenseLayer(int in_size, int out_size);
    std::vector<float> forward(const std::vector<float> &input);

private:
    std::vector<std::vector<float>> weights;
    std::vector<float> biases;
    int input_size, output_size;
};

#endif // DENSE_LAYER_H
