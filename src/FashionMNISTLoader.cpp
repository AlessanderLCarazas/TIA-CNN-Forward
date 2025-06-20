#include "FashionMNISTLoader.h"
#include <fstream>
#include <sstream>
#include <iostream>
#include <iomanip>

Tensor3D FashionMNISTSample::toTensor3D() const {
    Tensor3D tensor(28, 28, 1);
    for (int h = 0; h < 28; h++) {
        for (int w = 0; w < 28; w++) {
            int idx = h * 28 + w;
            tensor(h, w, 0) = pixels[idx] / 255.0f;
        }
    }
    return tensor;
}

bool FashionMNISTLoader::loadCSV(const std::string &filename) {
    std::ifstream file(filename);
    if (!file.is_open()) {
        std::cerr << "Error: No se puede abrir el archivo " << filename << std::endl;
        return false;
    }
    std::string line;
    bool is_header = true;
    int sample_count = 0;
    std::cout << "Cargando Fashion-MNIST dataset..." << std::endl;
    while (std::getline(file, line) && sample_count < 1000) {
        if (is_header) {
            is_header = false;
            continue;
        }
        std::stringstream ss(line);
        std::string cell;
        FashionMNISTSample sample;
        int column = 0;
        while (std::getline(ss, cell, ',')) {
            if (column == 0) {
                sample.label = std::stoi(cell);
            } else {
                sample.pixels.push_back(std::stof(cell));
            }
            column++;
        }
        if (sample.pixels.size() == 784) {
            dataset.push_back(sample);
            sample_count++;
            if (sample_count % 100 == 0) {
                std::cout << "Cargadas " << sample_count << " muestras..." << std::endl;
            }
        }
    }
    file.close();
    std::cout << "Dataset cargado: " << dataset.size() << " muestras" << std::endl;
    return !dataset.empty();
}

const std::vector<FashionMNISTSample>& FashionMNISTLoader::getDataset() const {
    return dataset;
}

size_t FashionMNISTLoader::size() const {
    return dataset.size();
}

const FashionMNISTSample& FashionMNISTLoader::getSample(size_t index) const {
    return dataset[index];
}

std::string FashionMNISTLoader::getClassName(int label) const {
    const std::vector<std::string> class_names = {
        "T-shirt/top", "Trouser", "Pullover", "Dress", "Coat",
        "Sandal", "Shirt", "Sneaker", "Bag", "Ankle boot"};
    if (label >= 0 && label < static_cast<int>(class_names.size())) {
        return class_names[label];
    }
    return "Unknown";
}
