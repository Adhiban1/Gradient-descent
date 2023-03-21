#include <iostream>
#include <cmath>

float f(float m, float x) {
    return m * x;
}

float loss(float y, float m, float x) {
    return pow((y - f(m, x)), 2);
}

float dm(float y, float m, float x) {
    return -2 * x * (y - m * x);
}

int main() {
    float x = 2, m = 0, y = 4, lr = 0.1;
    for (int i = 0; i < 10; i++) {
        m -= lr * dm(y, m, x);
        std::cout << "m: " << m << " | Loss: " << loss(y, m, x) << '\n';
    }
}