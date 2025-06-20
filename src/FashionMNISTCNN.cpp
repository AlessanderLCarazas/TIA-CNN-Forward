#include "FashionMNISTCNN.h"
#include <iostream>

std::vector<float> flatten(const Tensor3D &tensor) {
    std::vector<float> flattened;
    for (int h = 0; h < tensor.height; h++) {
        for (int w = 0; w < tensor.width; w++) {
            for (int c = 0; c < tensor.channels; c++) {
                flattened.push_back(tensor(h, w, c));
            }
        }
    }
    return flattened;
}

FashionMNISTCNN::FashionMNISTCNN() {
    std::cout << "\nINICIALIZANDO CNN PARA FASHION-MNIST" << std::endl;
    conv_layers.emplace_back(32, 3, 1, 1, 1);
    pool_layers.emplace_back(2, 2, MAX_POOLING);
    conv_layers.emplace_back(64, 3, 32, 1, 1);
    pool_layers.emplace_back(2, 2, MAX_POOLING);
    conv_layers.emplace_back(128, 3, 64, 1, 1);
    pool_layers.emplace_back(2, 2, AVERAGE_POOLING);
    dense_layers.emplace_back(6272, 256);
    dense_layers.emplace_back(256, 128);
    dense_layers.emplace_back(128, 10);
    std::cout << "CNN inicializada con arquitectura completa" << std::endl;
}

std::vector<float> FashionMNISTCNN::predict(const Tensor3D &input) {
    Tensor3D current_tensor = input;
    for (size_t i = 0; i < conv_layers.size(); i++) {
        current_tensor = conv_layers[i].forward(current_tensor);
        if (i < pool_layers.size()) {
            current_tensor = pool_layers[i].forward(current_tensor);
        }
    }
    std::vector<float> flattened = flatten(current_tensor);
    for (size_t i = 0; i < dense_layers.size(); i++) {
        flattened = dense_layers[i].forward(flattened);
    }
    return flattened;
}

int FashionMNISTCNN::classify(const Tensor3D &input) {
    std::vector<float> output = predict(input);
    int predicted_class = 0;
    float max_value = output[0];
    for (size_t i = 1; i < output.size(); i++) {
        if (output[i] > max_value) {
            max_value = output[i];
            predicted_class = static_cast<int>(i);
        }
    }
    return predicted_class;
}
