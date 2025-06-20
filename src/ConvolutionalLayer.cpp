#include "ConvolutionalLayer.h"
#include <random>
#include <algorithm>
#include "ActivationFunctions.h"

Tensor3D applyPadding(const Tensor3D &input, int padding_size) {
    int new_height = input.height + 2 * padding_size;
    int new_width = input.width + 2 * padding_size;
    Tensor3D padded(new_height, new_width, input.channels);
    for (int h = 0; h < input.height; h++) {
        for (int w = 0; w < input.width; w++) {
            for (int c = 0; c < input.channels; c++) {
                padded(h + padding_size, w + padding_size, c) = input(h, w, c);
            }
        }
    }
    return padded;
}

ConvolutionalLayer::ConvolutionalLayer(int num_filters, int kernel_size, int input_channels, int stride_val, int padding_val)
    : kernels(num_filters, kernel_size, kernel_size, input_channels), biases(num_filters, 0.0f), stride(stride_val), padding(padding_val) {
    std::random_device rd;
    std::mt19937 gen(rd());
    std::normal_distribution<float> dist(0.0f, 0.1f);
    for (int n = 0; n < num_filters; n++) {
        for (int h = 0; h < kernel_size; h++) {
            for (int w = 0; w < kernel_size; w++) {
                for (int c = 0; c < input_channels; c++) {
                    kernels(n, h, w, c) = dist(gen);
                }
            }
        }
    }
}

Tensor3D ConvolutionalLayer::forward(const Tensor3D &input) {
    Tensor3D processed_input = (padding > 0) ? applyPadding(input, padding) : input;
    int output_height = (processed_input.height - kernels.height) / stride + 1;
    int output_width = (processed_input.width - kernels.width) / stride + 1;
    Tensor3D output(output_height, output_width, kernels.num_kernels);
    for (int filter = 0; filter < kernels.num_kernels; filter++) {
        for (int out_h = 0; out_h < output_height; out_h++) {
            for (int out_w = 0; out_w < output_width; out_w++) {
                float convolution_sum = 0.0f;
                for (int k_h = 0; k_h < kernels.height; k_h++) {
                    for (int k_w = 0; k_w < kernels.width; k_w++) {
                        for (int c = 0; c < processed_input.channels; c++) {
                            int input_h = out_h * stride + k_h;
                            int input_w = out_w * stride + k_w;
                            convolution_sum += processed_input(input_h, input_w, c) * kernels(filter, k_h, k_w, c);
                        }
                    }
                }
                output(out_h, out_w, filter) = relu(convolution_sum + biases[filter]);
            }
        }
    }
    return output;
}
