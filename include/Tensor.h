#ifndef TENSOR_H
#define TENSOR_H

#include <vector>

class Tensor3D {
public:
    std::vector<std::vector<std::vector<float>>> data;
    int height, width, channels;
    Tensor3D(int h, int w, int c);
    float& operator()(int h, int w, int c);
    const float& operator()(int h, int w, int c) const;
};

class Tensor4D {
public:
    std::vector<std::vector<std::vector<std::vector<float>>>> data;
    int num_kernels, height, width, channels;
    Tensor4D(int n, int h, int w, int c);
    float& operator()(int n, int h, int w, int c);
    const float& operator()(int n, int h, int w, int c) const;
};

#endif // TENSOR_H
