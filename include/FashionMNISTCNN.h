#ifndef FASHION_MNIST_CNN_H
#define FASHION_MNIST_CNN_H

#include <vector>
#include "Tensor.h"
#include "ConvolutionalLayer.h"
#include "PoolingLayer.h"
#include "DenseLayer.h"

class FashionMNISTCNN {
public:
    FashionMNISTCNN();
    std::vector<float> predict(const Tensor3D &input);
    int classify(const Tensor3D &input);

private:
    std::vector<ConvolutionalLayer> conv_layers;
    std::vector<PoolingLayer> pool_layers;
    std::vector<DenseLayer> dense_layers;
};

std::vector<float> flatten(const Tensor3D &tensor);

#endif // FASHION_MNIST_CNN_H
