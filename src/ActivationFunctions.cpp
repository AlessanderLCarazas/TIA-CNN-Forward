#include "ActivationFunctions.h"
#include <algorithm>

float relu(float x) {
    return std::max(0.0f, x);
}
