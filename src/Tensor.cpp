#include "Tensor.h"

Tensor3D::Tensor3D(int h, int w, int c) : height(h), width(w), channels(c) {
    data.resize(h, std::vector<std::vector<float>>(w, std::vector<float>(c, 0.0f)));
}

float& Tensor3D::operator()(int h, int w, int c) {
    return data[h][w][c];
}

const float& Tensor3D::operator()(int h, int w, int c) const {
    return data[h][w][c];
}

Tensor4D::Tensor4D(int n, int h, int w, int c) : num_kernels(n), height(h), width(w), channels(c) {
    data.resize(n, std::vector<std::vector<std::vector<float>>>(h, std::vector<std::vector<float>>(w, std::vector<float>(c, 0.0f))));
}

float& Tensor4D::operator()(int n, int h, int w, int c) {
    return data[n][h][w][c];
}

const float& Tensor4D::operator()(int n, int h, int w, int c) const {
    return data[n][h][w][c];
}
