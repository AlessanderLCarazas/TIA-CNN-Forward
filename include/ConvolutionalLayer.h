#ifndef CONVOLUTIONAL_LAYER_H
#define CONVOLUTIONAL_LAYER_H

#include "Tensor.h"

class ConvolutionalLayer {
public:
    ConvolutionalLayer(int num_filters, int kernel_size, int input_channels, int stride_val = 1, int padding_val = 0);
    Tensor3D forward(const Tensor3D &input);

private:
    Tensor4D kernels;
    std::vector<float> biases;
    int stride;
    int padding;
};

Tensor3D applyPadding(const Tensor3D &input, int padding_size);

#endif // CONVOLUTIONAL_LAYER_H
