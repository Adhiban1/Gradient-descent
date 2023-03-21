# Gradient Descent in Python
## Intro
Let us take $x=2$, $y=4$, $m=0$

Let us take an equation $y=mx$

If you substitute $x$, $y$ in above equation. we get

$m=y/x$

$m=4/2$

$m=2$

Now using gradient descent we are going to get $m=2$

## Python Code
Initializing the values:
```python
x = 2
m = 0
y = 4
```
Functions:
```python
f = lambda m, x : m * x
loss = lambda y, m, x : (y - f(m, x)) ** 2
```
$loss = (y - f(m,x))^2$

$loss = (y - mx)^2$

${{\partial (loss)} \over \partial m} = 2(y - mx).(-x)$

${{\partial (loss)} \over \partial m} = -x(y - mx)$
```python
dm = lambda y, m, x : -2 * x * (y - m * x)
```
for loop:
```python
lr = 0.1 # Leaarning Rate
for _ in range(10):
    m -= lr * dm(y, m, x)
    print(f'm: {m} | Loss: {loss(y, m, x)}')
```
Full code:
```python
x = 2
m = 0
y = 4

f = lambda m, x : m * x
loss = lambda y, m, x : (y - f(m, x)) ** 2
dm = lambda y, m, x : -2 * x * (y - m * x)

lr = 0.1 # Leaarning Rate
for _ in range(10):
    m -= lr * dm(y, m, x)
    print(f'm: {m} | Loss: {loss(y, m, x)}')
```
Output:
```output
m: 1.6 | Loss: 0.6399999999999997
m: 1.92 | Loss: 0.025600000000000046
m: 1.984 | Loss: 0.001024000000000002
m: 1.9968 | Loss: 4.0960000000002345e-05
m: 1.99936 | Loss: 1.6383999999998664e-06
m: 1.999872 | Loss: 6.553599999990371e-08
m: 1.9999744 | Loss: 2.6214400000143383e-09
m: 1.99999488 | Loss: 1.0485759999875455e-10
m: 1.999998976 | Loss: 4.194303999222586e-12
m: 1.9999997952 | Loss: 1.6777216011442259e-13
```
**This is the simple code how Gradient Descent works.**

## C++
Code:
```c++
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
```
Run this code in terminal:
```bash
g++ main.cpp -o main.out
```
```bash
./main.out
```
Output:
```
m: 1.6 | Loss: 0.64
m: 1.92 | Loss: 0.0256
m: 1.984 | Loss: 0.001024
m: 1.9968 | Loss: 4.09614e-05
m: 1.99936 | Loss: 1.63858e-06
m: 1.99987 | Loss: 6.55675e-08
m: 1.99997 | Loss: 2.62759e-09
m: 1.99999 | Loss: 1.05103e-10
m: 2 | Loss: 4.60432e-12
m: 2 | Loss: 2.27374e-13
```