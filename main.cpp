#include <iostream>
#include <iomanip>
#include "FashionMNISTLoader.h"
#include "FashionMNISTCNN.h"

int main() {
    std::cout << "CNN FASHION-MNIST CLASSIFIER" << std::endl;
    std::cout << "1.1 Padding" << std::endl;
    std::cout << "1.2 Kernel" << std::endl;
    std::cout << "1.3 Convolution" << std::endl;
    std::cout << "1.4 Stride" << std::endl;
    std::cout << "1.5 ReLU activation" << std::endl;
    std::cout << "1.6 Pooling (Max, Min, Average)" << std::endl;
    std::cout << "2.1 Conv layers output connection to MLP" << std::endl;

    FashionMNISTLoader loader;
    std::string csv_file = "fashion-mnist_train.csv";
    if (!loader.loadCSV(csv_file)) {
        std::cerr << "Error: No se pudo cargar el dataset. Asegurate de que el archivo existe." << std::endl;
        std::cerr << "Tip: Coloca el archivo 'fashion-mnist_train.csv' en el mismo directorio que el ejecutable." << std::endl;
        return -1;
    }

    FashionMNISTCNN cnn;
    std::cout << "\nPROBANDO CNN CON MUESTRAS DE FASHION-MNIST:" << std::endl;
    std::cout << "================================================" << std::endl;
    int correct_predictions = 0;
    int total_predictions = std::min(100, static_cast<int>(loader.size()));
    for (int i = 0; i < total_predictions; i++) {
        const FashionMNISTSample &sample = loader.getSample(i);
        Tensor3D input_tensor = sample.toTensor3D();
        int predicted_class = cnn.classify(input_tensor);
        bool is_correct = (predicted_class == sample.label);
        std::cout << "============ MUESTRA " << (i + 1) << " ============" << std::endl;
        std::cout << "IMAGEN ORIGINAL:" << std::endl;
        for (int h = 0; h < 28; h++) {
            for (int w = 0; w < 28; w++) {
                int idx = h * 28 + w;
                std::cout << std::setw(3) << sample.pixels[idx] << " ";
            }
            std::cout << std::endl;
        }
        std::cout << "Real=" << loader.getClassName(sample.label) << " (" << sample.label << "), ";
        std::cout << "Prediccion=" << loader.getClassName(predicted_class) << " (" << predicted_class << ") ";
        std::cout << (is_correct ? "Correcto" : "Incorrecto") << std::endl;
        if (is_correct) correct_predictions++;
    }

    double accuracy = (double)correct_predictions / total_predictions * 100.0;
    std::cout << "\n+------------------------------------+" << std::endl;
    std::cout << "|         RESULTADOS FINALES:         |" << std::endl;
    std::cout << "+------------------------------------+" << std::endl;
    std::cout << "| Muestras procesadas: " << std::setw(14) << total_predictions << " |" << std::endl;
    std::cout << "| Predicciones correctas: " << std::setw(11) << correct_predictions << " |" << std::endl;
    std::cout << "| Precision: " << std::setw(23) << accuracy << "% |" << std::endl;
    std::cout << "+------------------------------------+" << std::endl;
    std::cout << "| FORWARD PASS COMPLETADO EXITOSAMENTE|" << std::endl;
    std::cout << "| La precision esperada es ~10%       |" << std::endl;
    std::cout << "| (aleatoria para 10 clases)          |" << std::endl;
    std::cout << "+------------------------------------+" << std::endl;

    return 0;
}
