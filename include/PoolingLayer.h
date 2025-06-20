#ifndef POOLING_LAYER_H
#define POOLING_LAYER_H

#include "Tensor.h"

enum PoolingType {
    MAX_POOLING,
    MIN_POOLING,
    AVERAGE_POOLING
};

class PoolingLayer {
public:
    PoolingLayer(int pool_sz, int stride_val, PoolingType type);
    Tensor3D forward(const Tensor3D &input);

private:
    int pool_size;
    int stride;
    PoolingType pooling_type;
};

#endif // POOLING_LAYER_H
