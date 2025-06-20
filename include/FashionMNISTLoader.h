#ifndef FASHION_MNIST_LOADER_H
#define FASHION_MNIST_LOADER_H

#include <vector>
#include <string>
#include "Tensor.h"

struct FashionMNISTSample {
    int label;
    std::vector<float> pixels;
    Tensor3D toTensor3D() const;
};

class FashionMNISTLoader {
public:
    bool loadCSV(const std::string &filename);
    const std::vector<FashionMNISTSample>& getDataset() const;
    size_t size() const;
    const FashionMNISTSample& getSample(size_t index) const;
    std::string getClassName(int label) const;

private:
    std::vector<FashionMNISTSample> dataset;
};

#endif // FASHION_MNIST_LOADER_H
