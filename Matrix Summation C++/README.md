# Gradient Descent
## Formula
- $A$ - input matrix $(a_{ij})$
- $W$ - Weight matrix $(w_{j1})$
- $Y$ - True output matrix $(y_{i1})$
- $\hat{Y}$ - Predicted matrix $(\hat{y}_{i1})$

$$\displaystyle A = \begin{bmatrix}a_{11}&a_{12}&\dots&a_{1j} \\ 
a_{21}&a_{22}&\dots&a_{2j}\\ 
\vdots&\vdots&\ddots&\vdots \\ 
a_{i1}&a_{i2}&\dots&a_{ij}\end{bmatrix} \Rightarrow A = a_{ij}$$

$$\displaystyle W = \begin{bmatrix}w_{11}\\
w_{21}\\
\vdots\\
w_{j1}\end{bmatrix} \Rightarrow W = w_{j1}$$

$$\displaystyle Y = \begin{bmatrix}y_{11}\\
y_{21}\\
\vdots\\
y_{i1}\end{bmatrix} \Rightarrow Y=y_{i1}$$

$$\displaystyle \hat{Y} = \begin{bmatrix}\hat{y}_{11}\\
\hat{y}_{21}\\
\vdots\\
\hat{y}_{i1}\end{bmatrix} \Rightarrow \hat{Y}=\hat{y}_{i1}$$

**Mean Square Error:**

$\displaystyle loss = {1 \over N}\sum\left(Y-\hat{Y}\right)^2 \Rightarrow loss={1 \over N_i}\sum_{m=1}^{i}\left(y_{m1}-\sum_{n=1}^{j} a_{mn}w_{n1}\right)^2$

**Gradient:**

$\displaystyle {d(loss) \over d(w_{j1})}={d \over d(w_{j1})}\left[{1 \over N_i}\sum_{m=1}^{i}\left(y_{m1}-\sum_{n=1}^{j} a_{mn}w_{n1}\right)^2\right]$

$\displaystyle {d(loss) \over d(w_{j1})}={1 \over N_i}\sum_{m=1}^{i}\left[2\left(y_{m1}-\sum_{n=1}^{j} a_{mn}w_{n1}\right).\left(-{d \over d(w_{j1})}\sum_{n=1}^{j} a_{mn}w_{n1}\right)\right]$

$\displaystyle {d(loss) \over d(w_{j1})}={1 \over N_i}\sum_{m=1}^{i}\left[2\left(y_{m1}-\sum_{n=1}^{j} a_{mn}w_{n1}\right).\left(-a_{mj}\right)\right]$

$\displaystyle {d(loss) \over d(w_{j1})}=-{2 \over N_i}\sum_{m=1}^{i}\left[\left(a_{mj}\right)\left(y_{m1}-\sum_{n=1}^{j} a_{mn}w_{n1}\right)\right]$

## Code
***I used C++, we can use Python also, but using this in Python is slow, for Python `Numpy` is the best***

Matrices
```c++
array<array<double, 4>, 3> A = {{{1,1,0,0},{0,0,0,0},{1,1,1,1}}};
    array<array<double, 1>, 4> W = {{{0},{0},{0},{0}}};
    array<array<double, 1>, 3> Y = {{{2},{0},{4}}};
```

Loss
```c++
double loss(
    array<array<double, 4>, 3> A, 
    array<array<double, 1>, 4> W, 
    array<array<double, 1>, 3> Y){
    double l = 0;
    for(int i=0; i<A.size(); i++){
        double yhat = 0;
        for(int j=0; j<A[0].size(); j++){
            yhat += A[i][j]*W[j][0];
        }
        l += pow((Y[i][0] - yhat), 2);
    }
    l = l/A.size();
    return l;
}
```

Gradient
```c++
array<array<double, 1>, 4> grad(
    array<array<double, 4>, 3> A, 
    array<array<double, 1>, 4> W, 
    array<array<double, 1>, 3> Y, 
    double lr){

    array<array<double, 1>, 4> dw;
    for(int j=0; j<A[0].size(); j++){
        dw[j][0] = 0;
        for(int i=0; i<A.size(); i++){
            double yhat = 0;
            for(int j1=0; j1<A[0].size(); j1++){
                yhat += A[i][j1]*W[j1][0];
            }
            dw[j][0] += A[i][j]*(Y[i][0] - yhat);
        }
        dw[j][0] = dw[j][0] * (-2) / A.size();
    }
    
    for(int j=0; j<W.size(); j++){
        W[j][0] -= lr*dw[j][0];
    }
    return W;
}
```

`result` 

$$\displaystyle \hat{Y}=A.W \Rightarrow \hat{y}_{i1}=\sum_{n=1}^{j} a_{in}w_{n1}$$

```c++
array<array<double, 1>, 3> result(array<array<double, 4>, 3> A, array<array<double, 1>, 4> W){
    array<array<double, 1>, 3> y;
    for(int i=0; i<A.size(); i++){
        double temp = 0;
        for(int j=0; j<A[0].size(); j++){
            temp += A[i][j]*W[j][0];
        }
        y[i][0] = temp;
    }
    return y;
}
```

iteration
```c++
int epochs = 1000;
for(int i=1; i<=epochs; i++){
    W = grad(A,W,Y, 0.1);
    if(i%(epochs/10) == 0)
    cout<<"Loss: "<<loss(A, W, Y)<<'\n';
}

for(auto i : result(A,W)){
    for(auto j : i){
        cout<<j<<' ';
    }
    cout<<'\n';
}
```

Compile
```c++
g++ main.cpp -o main.out
```

Run
```c++
./main.out
```
Output
```c++
Loss: 1.5498e-06
Loss: 4.46718e-11
Loss: 1.28763e-15
Loss: 3.71151e-20
Loss: 1.06976e-24
Loss: 3.16859e-29
Loss: 6.57384e-31
Loss: 6.57384e-31
Loss: 6.57384e-31
Loss: 6.57384e-31
2 
0 
4
```
