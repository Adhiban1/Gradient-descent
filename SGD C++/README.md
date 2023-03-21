# Stochastic Gradient Descent
## Formula
Input:

$$
A = \left(\begin{array}{cc}
a_{11} & a_{12} & ... & a_{1j}\\
a_{21} & a_{22} & ... & a_{2j}\\
. & . & & .\\
. & . & & .\\
a_{i1} & a_{i2} & ... & a_{ij}\\
\end{array}\right)
$$

Weight:

$$
W = \left(\begin{array}{cc}
w_1\\
w_2\\
.\\
.\\
w_j
\end{array}\right)
$$

Bias: 

$$b$$

Y:

$$
Y = \left(\begin{array}{cc}
y_1\\
y_2\\
.\\
.\\
y_i
\end{array}\right)
$$

Y_pred:

$$
Y\_pred = A.W+b
$$

$$
Y\_pred = \left(\begin{array}{cc}
a_{11} & a_{12} & ... & a_{1j}\\
a_{21} & a_{22} & ... & a_{2j}\\
. & . & & .\\
. & . & & .\\
a_{i1} & a_{i2} & ... & a_{ij}\\
\end{array}\right).
\left(\begin{array}{cc}
w_1\\
w_2\\
.\\
.\\
w_j
\end{array}\right)+b
$$

$$
Y\_pred = \left(\begin{array}{cc}
a_{11}w_{1}+a_{12}w_2+...+a_{1j}w_j+b\\
a_{21}w_{1}+a_{22}w_2+...+a_{2j}w_j+b\\
.\\
.\\
a_{i1}w_{1}+a_{i2}w_2+...+a_{ij}w_j+b
\end{array}\right)
$$

loss:

$$
loss = {1 \over N}\sum(Y-Y\_pred)^2
$$

$$
loss={1 \over N}\left[[y_1-(a_{11}w_{1}+a_{12}w_2+...+a_{1j}w_j+b)]^2\\
+[y_2-(a_{21}w_{1}+a_{22}w_2+...+a_{2j}w_j+b)]^2\\
+...+[y_i-(a_{i1}w_{1}+a_{i2}w_2+...+a_{ij}w_j+b)]^2\right]
$$

W grad:

$$
{d(loss) \over d(w_j)} = {d \over d(w_j)}[(Y-Y\_pred)^2]
$$

$$
{d(loss) \over d(w_j)} = {d \over d(w_j)}\{[y_i-(a_{i1}w_{1}+a_{i2}w_2+...+a_{ij}w_j+b)]^2\}
$$

$$
{d(loss) \over d(w_j)} = -2a_{ij}[y_i-(a_{i1}w_{1}+a_{i2}w_2+...+a_{ij}w_j+b)]
$$

Bias grad:

$$
{d(loss) \over d(b)} = -2[y_i-(a_{i1}w_{1}+a_{i2}w_2+...+a_{ij}w_j+b)]
$$

## code
```c++
#include<iostream>
#include<vector>
#include<cmath>
using namespace std;
typedef vector<vector<double>> Matrix;
```
Y_pred
```c++
vector<double> y_pred(Matrix a, vector<double> w, double b){
    vector<double> c{};
    for(int i=0; i<a.size(); i++){
        double temp=0;
        for(int j=0; j<a[i].size(); j++){
            temp += a[i][j]*w[j];
        }
        temp += b;
        c.push_back(temp);
    }
    return c;
}
```
Loss
```c++
double loss(Matrix a, vector<double> w, vector<double> y, double b){
    double l=0;
    for(int i=0; i<a.size(); i++){
        double temp=0;
        for(int j=0; j<a[0].size(); j++){
            temp -= a[i][j]*w[j];
        }
        l += pow((y[i]+temp-b),2);
    }
    l = l/a.size();
    return l;
}
```
Update
```c++
void update_grad(Matrix a, vector<double> &w, vector<double> y, double &b, int epochs, double lr){
    for(int k=0; k<epochs; k++){
        for(int i=0; i<a.size(); i++){
            double temp=0;
            for(int j=0; j<a[0].size(); j++){
                temp += -a[i][j]*w[j];
            }
            temp = (y[i]+temp-b);
            for(int j=0; j<a[0].size(); j++){
                w[j] -= -lr*2*a[i][j]*(temp);
            }
            b -= -2*lr*(temp);
        }
        if(k > epochs-10)
        cout<<"Loss: "<<loss(a, w, y, b)<<'\n';
    }
}
```
main
```c++
int main(){
    Matrix a = {
        {0,0},
        {0,1},
        {1,0},
        {1,1}
    };
    vector<double> y = {0,1,1,2};
    vector<double> w = {0,0};
    double b = 0;
    update_grad(a, w, y, b, 1000, 0.01);
    vector<double> predict = y_pred(a, w, b);

    cout<<"\nY:         ";
    for(double i:y){
        cout<<i<<' ';
    }
    cout<<'\n';

    cout<<"Y-Predict: ";
    for(double i:predict){
        cout<<i<<' ';
    }
    cout<<'\n';
}
```
Run the code
```bash
g++ main.cpp -o main.out
```
```bash
./main.out
```
Output
```c++
Loss: 1.87316e-12
Loss: 1.82594e-12
Loss: 1.77991e-12
Loss: 1.73505e-12
Loss: 1.69131e-12
Loss: 1.64867e-12
Loss: 1.60711e-12
Loss: 1.5666e-12
Loss: 1.52711e-12

Y:         0 1 1 2 
Y-Predict: 2.03214e-06 1 1 2
```