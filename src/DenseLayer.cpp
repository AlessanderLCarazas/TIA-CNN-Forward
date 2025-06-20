#include "DenseLayer.h"
#include "ActivationFunctions.h"
#include <random>

DenseLayer::DenseLayer(int in_size, int out_size) : input_size(in_size), output_size(out_size) {
    weights.resize(output_size, std::vector<float>(input_size));
    biases.resize(output_size);
    std::random_device rd;
    std::mt19937 gen(rd());
    std::normal_distribution<float> dist(0.0f, 0.1f);
    for (int i = 0; i < output_size; i++) {
        for (int j = 0; j < input_size; j++) {
            weights[i][j] = dist(gen);
        }
        biases[i] = dist(gen);
    }
}

std::vector<float> DenseLayer::forward(const std::vector<float> &input) {
    std::vector<float> output(output_size);
    for (int i = 0; i < output_size; i++) {
        float sum = 0.0f;
        for (int j = 0; j < input_size; j++) {
            sum += weights[i][j] * input[j];
        }
        output[i] = relu(sum + biases[i]);
    }
    return output;
}
