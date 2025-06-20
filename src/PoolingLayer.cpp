#include "PoolingLayer.h"
#include <algorithm>

PoolingLayer::PoolingLayer(int pool_sz, int stride_val, PoolingType type)
    : pool_size(pool_sz), stride(stride_val), pooling_type(type) {}

Tensor3D PoolingLayer::forward(const Tensor3D &input) {
    int output_height = (input.height - pool_size) / stride + 1;
    int output_width = (input.width - pool_size) / stride + 1;
    Tensor3D output(output_height, output_width, input.channels);
    for (int c = 0; c < input.channels; c++) {
        for (int out_h = 0; out_h < output_height; out_h++) {
            for (int out_w = 0; out_w < output_width; out_w++) {
                float pool_result;
                bool first_value = true;
                float sum_for_average = 0.0f;
                for (int p_h = 0; p_h < pool_size; p_h++) {
                    for (int p_w = 0; p_w < pool_size; p_w++) {
                        int input_h = out_h * stride + p_h;
                        int input_w = out_w * stride + p_w;
                        if (input_h < input.height && input_w < input.width) {
                            float current_value = input(input_h, input_w, c);
                            if (first_value) {
                                pool_result = current_value;
                                first_value = false;
                            }
                            switch (pooling_type) {
                            case MAX_POOLING:
                                pool_result = std::max(pool_result, current_value);
                                break;
                            case MIN_POOLING:
                                pool_result = std::min(pool_result, current_value);
                                break;
                            case AVERAGE_POOLING:
                                sum_for_average += current_value;
                                break;
                            }
                        }
                    }
                }
                if (pooling_type == AVERAGE_POOLING) {
                    pool_result = sum_for_average / (pool_size * pool_size);
                }
                output(out_h, out_w, c) = pool_result;
            }
        }
    }
    return output;
}
