# 4 Points Python-C++
## Formula
The 3rd order equation we have to find out is $\displaystyle y=ax^3+bx^2+cx+d$

$$\displaystyle A= \begin{bmatrix}x_1^3&x_1^2&x_1&1\\
x_2^3&x_2^2&x_2&1\\
x_3^3&x_3^2&x_3&1\\
x_4^3&x_4^2&x_4&1\\
\end{bmatrix}$$

$$\displaystyle W = \begin{bmatrix}a\\
b\\
c\\
d
\end{bmatrix}$$

$$\displaystyle Y = \begin{bmatrix}y_1\\
y_2\\
y_3\\
y_4
\end{bmatrix}$$

$\hat{Y} = A \cdot W$

$\displaystyle loss = \frac{1}{N}\sum \left(Y-\hat{Y}\right)^2 \Rightarrow loss = \frac{1}{N}\sum_{i=1}^{m}\left(y_{i1}-\sum_{j=1}^{n}a_{ij}w_{j1}\right)^2$

$\displaystyle \frac{d(loss)}{d(w_{k1})} = -\frac{2}{N}\sum_{i=1}^{m}\left[\left(y_{i1}-\sum_{j=1}^{n}a_{ij}w_{j1}\right)(a_{ik})\right]$

## C++ code
> I'm using C++ to speedup python

**Import libraries**
```c++
#include <iostream>
#include <array>
#include <cmath>
```
**Matrices**
```c++
array<array<double, 4>, 4> A;
array<array<double, 1>, 4> W{{{0}, {0}, {0}, {0}}};
array<array<double, 1>, 4> Y;
```
**points**
```c++
double points[4][2] = {
    {py_points[0], py_points[1]},
    {py_points[2], py_points[3]},
    {py_points[4], py_points[5]},
    {py_points[6], py_points[7]}};
```
> Changing 1d to 2d array

**Seperate x and y from points**
```c++
double x[4], y[4];
for (int i = 0; i < 4; i++)
{
    x[i] = points[i][0];
    y[i] = points[i][1];
}
```
**Initializing matrices**
```c++
array<array<double, 4>, 4> A;
array<array<double, 1>, 4> W{{{0}, {0}, {0}, {0}}};
array<array<double, 1>, 4> Y;
```
**Assigning values to matrices**
```c++
for (int i = 0; i < 4; i++)
{
    A[i] = {pow(x[i], 3), pow(x[i], 2), x[i], 1};
}
for (int i = 0; i < 4; i++)
{
    Y[i][0] = y[i];
}
```
**loss (mean square error)**
```c++
double loss(
    array<array<double, 4>, 4> A,
    array<array<double, 1>, 4> W,
    array<array<double, 1>, 4> Y)
{
    double l = 0;
    for (int i = 0; i < 4; i++)
    {
        double temp = 0;
        for (int j = 0; j < 4; j++)
        {
            temp += A[i][j] * W[j][0];
        }
        l += pow((Y[i][0] - temp), 2);
    }
    l = l / 4;
    return l;
}
```
**grad function**
```c++
array<array<double, 1>, 4> grad(
    array<array<double, 4>, 4> A,
    array<array<double, 1>, 4> W,
    array<array<double, 1>, 4> Y,
    double lr, int epochs)
{
    for (int m = 0; m < epochs; m++)
    {
        for (int k = 0; k < 4; k++)
        {
            double temp1 = 0;
            for (int i = 0; i < 4; i++)
            {
                double temp2 = 0;
                for (int l = 0; l < 4; l++)
                {
                    temp2 += A[i][l] * W[l][0];
                }
                temp1 += (Y[i][0] - temp2) * (A[i][k]);
            }
            W[k][0] = W[k][0] + lr * 2 * temp1;
        }
        cout << "Loss: " << loss(A, W, Y) << '\n';
    }
    return W;
}
```
**main function**
```c++
extern "C"
{
    void function(double *py_points, double* dw, int epochs, double lr)
    {

        // double points[4][2] = {{-1, 1}, {0, 0}, {1, 1}, {2, 0}};
        double points[4][2] = {
            {py_points[0], py_points[1]},
            {py_points[2], py_points[3]},
            {py_points[4], py_points[5]},
            {py_points[6], py_points[7]}};
        double x[4], y[4];

        for (int i = 0; i < 4; i++)
        {
            x[i] = points[i][0];
            y[i] = points[i][1];
        }

        array<array<double, 4>, 4> A;
        array<array<double, 1>, 4> W{{{0}, {0}, {0}, {0}}};
        array<array<double, 1>, 4> Y;

        for (int i = 0; i < 4; i++)
        {
            A[i] = {pow(x[i], 3), pow(x[i], 2), x[i], 1};
        }
        for (int i = 0; i < 4; i++)
        {
            Y[i][0] = y[i];
        }

        W = grad(A, W, Y, lr, epochs);

        for (int i = 0; i < 4; i++)
        {
            dw[i] = W[i][0];
        }

        dw[4] = loss(A, W, Y);

        cout << "W\n";

        for (int i=0; i<5; i++)
        {
            cout << dw[i] << ' ';
        }
        cout << '\n';
    }
}
```
**compiling C++**
```bash
g++ main.cpp -shared -fPIC -o grad.so
```
## Python code
**Import packages**
```python
import ctypes
import numpy as np
import plotly.express as px
import plotly.graph_objs as go
from flask import Flask, render_template, request
import re
```
$\displaystyle y=ax^3+bx^2+cx+d$
```python
def func(x, w):
    return w[0]*x**3+w[1]*x**2+w[2]*x+w[3]
```
**using C++ in python**
```c++
clib = ctypes.CDLL('./grad.so')
clib.function.argtypes = [ctypes.POINTER(ctypes.c_double), 
                          ctypes.POINTER(ctypes.c_double), 
                          ctypes.c_int, ctypes.c_double]
clib.function.restype = ctypes.POINTER(ctypes.c_double)
```
**Flask app**
```python
app = Flask(__name__)

@app.route('/', methods=['POST', 'GET'])
def home():
    fig = px.line(x=[0], y=[0])
    loss = 'NaN'
    equ = ''
    if request.method == 'POST':
        w = (ctypes.c_double*5)(0,0,0,0,0)
        # arr = [-1, 1, 0, 0, 1, 1, 2, 0]
        arr = list(map(float, re.findall('[-\d\.]+', request.form.get('points'))))

        x1 = [arr[i*2] for i in range(4)]
        y1 = [arr[i*2+1] for i in range(4)]

        clib.function(
            (ctypes.c_double*8)(*arr),
            w,
            1000,
            0.01
        )

        loss = w[4]

        x = np.linspace(min(x1)-1, max(x1)+1, 100)
        y = func(x, w)

        # plt.plot(x, y)
        # plt.plot(x1, y1, 'o', color='red')
        # plt.title(f'${w[0]:.2f}x^3+{w[1]:.2f}x^2+{w[2]:.2f}x+{w[3]:.2f}$')
        # plt.show()
        fig1 = px.line(x=x, y=y)
        fig2 = px.scatter(x=x1, y=y1)
        fig = go.Figure(data=[fig1.data[0], fig2.data[0]])
        fig.data[1].marker.color = '#ff0000'
        fig.update_layout(
            xaxis_title="X Axis",
            yaxis_title="Y Axis"
        )
        equ = f'\\({w[0]:.2f}x^3+{w[1]:.2f}x^2+{w[2]:.2f}x+{w[3]:.2f}\\)'
    return render_template('index.html', graph=fig.to_json(), loss=loss, 
                           equ=equ)

if __name__ == '__main__':
    app.run(debug=True)
```
## Html code
```html
<!DOCTYPE html>
<html lang="en">
<head>
    <style>
        th{
            text-align: right;
        }
        th,td{
            padding: 10px;
        }
        body{
            margin: 25px;
            font-size: 150%;
        }
        input{
            font-size: 100%;
            padding: 5px;
            margin: 5px;
        }
        #graph{
            margin-left: 50px;
            margin-right: 50px;
            height: 800px;
        }
    </style>
    <script src="https://polyfill.io/v3/polyfill.min.js?features=es6"></script>
    <script type="text/javascript" id="MathJax-script" async
      src="https://cdn.jsdelivr.net/npm/mathjax@3/es5/tex-chtml.js">
    </script>
    <script src="https://cdn.plot.ly/plotly-latest.min.js"></script>
    <meta charset="UTF-8">
    <meta http-equiv="X-UA-Compatible" content="IE=edge">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Document</title>
</head>
<body>
    <center>
        <h1>Gradent Descent Python & C++</h1>
    </center>
    <form action="/" method="POST">
        <input type="text" name="points" placeholder="points">
        <input type="submit">
    </form>
    <table>
        <tr><th>Loss:</th><td>{{loss}}</td></tr>
        <tr><th>Equation:</th><td>{{equ}}</td></tr>
    </table>
    <div id="graph"></div>
    <script>
        Plotly.plot('graph', {{graph | safe}});
    </script>
</body>
</html>
```