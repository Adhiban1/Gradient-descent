# Gradient Descent
Let us take
- Matrix $X$ of shape $10$ X $1$
- Matrix $W$ of shape $1$ X $10$
- $B$ is a number
- Output Matrix $Y=W.X+B$ ..........(1)

Let us take Matrix $w$ contains random elements and random number $b$.

Now, we are going to use **Gradient Descent** to find $w$ and $b$ that satisfied equation no. (1).
## Code
Importing packages
```python
import numpy as np
import matplotlib.pyplot as plt
```
Initializing values
```python
np.random.seed(0)

X = np.random.randint(0, 101, (10, 1))
W = np.random.randint(0, 101, (1, 10))
B = np.random.randint(0, 101)
Y = W @ X + B
```
Let us take Matrix $w$ contains random elements and random number $b$.
```python
w = np.random.randint(0, 101, (1, 10))
b = np.random.randint(0, 101)
```
**Loss function:**

$loss={1 \over N}{\sum (Y - w.X - b)^2}$
```python
loss = ((Y - w @ X - b)**2).mean()
```
**Gradient:**

${\partial (loss) \over \partial w}=-2X^T(Y-w.X-b)$

${\partial (loss) \over \partial b}=-2(Y-w.X-b)$
```python
def dw(X, Y, w, b):
    return -2*X.T*(Y - w @ X - b)

def db(X, Y, w, b):
    return -2*(Y - w @ X - b)
```
for loop
```python
lr = 1e-6
d = []
for i in range(100):
    w = w - lr*dw(X, Y, w, b)
    b = b - lr*db(X, Y, w, b)

    loss = ((Y - w @ X - b)**2).mean()

    d.append(loss)
print(W @ X + B, w @ X + b, sep='\n')
```
Loss graph
```python
plt.plot(d)
plt.grid()
plt.ylabel('Loss')
plt.savefig('loss.svg')
plt.close()
```
**Graph:**

<img src="https://github.com/Adhiban1/Data-Science/raw/main/gradient%20descent-python-c%2B%2B/2/loss.png">

---

# C++
## matrix.h
### Importing libraries
```c++
#include <iostream>
#include <vector>
```
### random number function
```c++
float randNumber(int a, int b)
{
    return rand() % (b - a + 1) + a;
}
```
This function takes starting and ending integer number `[a,b]` and return random number in between a and b including a and b.
### Matrix class
#### initializing variables
```c++
public:
    std::vector<std::vector<float>> matrix;
    int rows, columns;
    std::string shape;
```
#### constructor
```c++
Matrix(int a, int b)
    {
        rows = a;
        columns = b;
        for (int i = 0; i < a; i++)
        {
            std::vector<float> temp;
            for (int j = 0; j < b; j++)
            {
                temp.push_back(0);
            }
            matrix.push_back(temp);
        }
        shape = "Shape(" + std::to_string(rows) + "," + std::to_string(columns) + ")";
    }
```
Input parameters:
- a - Rows
- b - Columns

`Matrix A(10,5)` - creates matrix of shape (10,5)
#### operator+
```c++
Matrix operator+(Matrix other)
    {
        Matrix temp(rows, columns);
        for (int i = 0; i < rows; i++)
            for (int j = 0; j < columns; j++)
                temp.matrix[i][j] = matrix[i][j] + other.matrix[i][j];
        return temp;
    }
```
This will returns sum of two matrix, `Matrix C = A+B`
#### Matrix dot product
```c++
Matrix dot(Matrix other)
    {
        Matrix temp(rows, other.columns);
        for (int i = 0; i < rows; i++)
        {
            for (int k = 0; k < other.columns; k++)
            {
                float element = 0;
                for (int j = 0; j < columns; j++)
                {
                    element += matrix[i][j] * other.matrix[j][k];
                }
                temp.matrix[i][k] = element;
            }
        }
        return temp;
    }
```
`Matrix C = A.dot(B)` this will returns matrix dot product of $C = A.B$
#### transpose
```c++
Matrix T(){
        Matrix temp(columns, rows);
        for(int i = 0; i < rows; i++)
            for(int j = 0; j < columns; j++)
                temp.matrix[j][i] = matrix[i][j];
        return temp;
    }
```
`Matrix B = A.T()` this will return transpose of this matrix.
#### sum
```c++
float sum() {
        float temp = 0;
        for (int i = 0; i < rows; i++)
            for (int j = 0; j < columns; j++)
                temp += matrix[i][j];
        return temp;
    }
```
`float s = A.sum()` this will returns sum of all elements in this Matrix
#### mean
```c++
float mean() {
        return sum() / (rows + columns);
    }
```
`float s = A.mean()` this will returns mean of all elements in this Matrix
#### randfill
```c++
void randfill(int s, int e)
    {
        for (int i = 0; i < rows; i++)
            for (int j = 0; j < columns; j++)
                matrix[i][j] = randNumber(s, e);
    }
```
`A.randfill()` will fill random elements in matrix A.
#### print
```c++
void print(){
        for (int i = 0; i < rows; i++){
            for (int j = 0; j < columns; j++) {
                std::cout<<matrix[i][j]<<' ';
            }
            std::cout<<'\n';
        }
        std::cout<<"Shape("<<rows<<","<<columns<<")\n";
    }
```
`A.print()` will print the matrix.
## main.cpp
### Importing libraries
```c++
#include "matrix.h"
#include <cmath>
```
clear screen, random seed
```c++
system("clear"); // Linux
// system("cls") // Windows
srand(1);
```
### Initializing Matrix & float and other code
```c++
Matrix X(10, 1), W(1, 10), w(1, 10);
    X.randfill(0, 100);
    W.randfill(0, 100);
    w.randfill(0, 100);

    float B = randNumber(0, 100), b = randNumber(0, 100);
```
Y value
```c++
float Y = (W.dot(X) + B).mean();
```
loss value
```c++
float loss = pow((Y - w.dot(X).mean() - b), 2);
```
loop
```c++
float lr = 1e-5;
    for(int i = 0; i < 55; i++){
        w = w - dw(X, Y, w, b)*lr;
        b = b - db(X, Y, w, b)*lr;
        float l = pow((Y - w.dot(X).mean() - b), 2);
        std::cout<<i<<". Loss: "<<l<<'\n';
    }
    std::cout<<"y true:      "<<W.dot(X).mean() + B<<'\n'<<"Predicted y: "<<w.dot(X).mean() + b<<'\n';
```
### Run main.cpp
```c++
g++ main.cpp -o main.out
```
```c++
./main.out
```
### Output
```c++
0. Loss: 1.26292e+07
1. Loss: 7.54914e+06
2. Loss: 4.51252e+06
3. Loss: 2.69737e+06
4. Loss: 1.61236e+06
5. Loss: 963792
6. Loss: 576109
7. Loss: 344370
8. Loss: 205848
9. Loss: 123046
10. Loss: 73551.2
11. Loss: 43965.5
12. Loss: 26280.2
13. Loss: 15709.6
14. Loss: 9390.05
15. Loss: 5613.05
16. Loss: 3355.19
17. Loss: 2005.56
18. Loss: 1198.86
19. Loss: 716.557
20. Loss: 428.321
21. Loss: 256.071
22. Loss: 153.069
23. Loss: 91.513
24. Loss: 54.6745
25. Loss: 32.6881
26. Loss: 19.5484
27. Loss: 11.6785
28. Loss: 6.98502
29. Loss: 4.17096
30. Loss: 2.49441
31. Loss: 1.49071
32. Loss: 0.890352
33. Loss: 0.532465
34. Loss: 0.317729
35. Loss: 0.189866
36. Loss: 0.114293
37. Loss: 0.068078
38. Loss: 0.0409334
39. Loss: 0.024162
40. Loss: 0.014468
41. Loss: 0.00863737
42. Loss: 0.00510527
43. Loss: 0.00300827
44. Loss: 0.00186011
45. Loss: 0.0010489
46. Loss: 0.000703703
47. Loss: 0.000387754
48. Loss: 0.000249173
49. Loss: 0.000165265
50. Loss: 8.00896e-05
51. Loss: 4.89462e-05
52. Loss: 2.54321e-05
53. Loss: 1.65362e-05
54. Loss: 1.65362e-05
y true:      10757
Predicted y: 10731.5
```
