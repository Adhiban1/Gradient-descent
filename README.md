# 4points-python
Finding the values of coefficients of 3rd degree equation that passes through the 4 points that we have given. This is achived by using gradient descent and increasing learing rate exponentially (to decrease the number of iteration and get faster training).

$y = ax^3+bx^2+cx+d$, we have to find the $a, b, c, d$

Here 4 variables are there, so we should have 4 equations to solve it, to get 4 equations, we need 4 points and substituting these points on the above equation to get 4 equations.

Let us take 4 points $(x_1, y_1), (x_2, y_2), (x_3, y_3), (x_4, y_4)$. Substitute these points in above equation. We get,

- $y_1 = ax_1^3+bx_1^2+cx_1+d$
- $y_2 = ax_2^3+bx_2^2+cx_2+d$
- $y_3 = ax_3^3+bx_3^2+cx_3+d$
- $y_4 = ax_4^3+bx_4^2+cx_4+d$

We can convert this equation into Matrix,

$$
\begin{bmatrix}
y_1\\
y_2\\
y_3\\
y_4
\end{bmatrix}  = 
\begin{bmatrix}
x_1^3 & x_1^2 & x_1 & 1\\
x_2^3 & x_2^2 & x_2 & 1\\
x_3^3 & x_3^2 & x_3 & 1\\
x_4^3 & x_4^2 & x_4 & 1\\
\end{bmatrix}.
\begin{bmatrix}
a\\
b\\
c\\
d
\end{bmatrix}\\
$$

$$
Y = A.W
$$

### Loss
Let us take $Y$ as true value and $\hat{Y}$ as predicted value.

$$loss = {1 \over N}\sum(Y-\hat{Y})^2$$

### Gradient

$$
W' = {1 \over 2}\begin{bmatrix}
A_{col1}^T.(Y-\hat{Y})\\
A_{col2}^T.(Y-\hat{Y})\\
A_{col3}^T.(Y-\hat{Y})\\
A_{col4}^T.(Y-\hat{Y})\\
\end{bmatrix}
$$

# $y=mx$
Finding $m$ value when $x$ and $y$ values are given. Here simple Gradient descent is applied to find $w$

$${{\partial (loss)} \over \partial m} = -x(y - mx)$$

> Using this we can learn how gradient descent works.

# $Y=W.X+B$
- Matrix $X$ of shape $10$ X $1$
- Matrix $W$ of shape $1$ X $10$
- $B$ is a number
- Output Matrix $Y=W.X+B \dots (1)$

**Loss function:**

$\displaystyle loss={1 \over N}{\sum (Y - w.X - b)^2}$

**Gradient:**

$\displaystyle {\partial (loss) \over \partial w}=-2X^T(Y-w.X-b)$

$\displaystyle {\partial (loss) \over \partial b}=-2(Y-w.X-b)$

# Stochastic Gradient Descent C++
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
{d(loss) \over d(w_j)} = -2a_{ij}[y_i-(a_{i1}w_{1}+a_{i2}w_2+...+a_{ij}w_j+b)]
$$

Bias grad:

$$
{d(loss) \over d(b)} = -2[y_i-(a_{i1}w_{1}+a_{i2}w_2+...+a_{ij}w_j+b)]
$$

# Matrix Summation C++
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

$\displaystyle loss={1 \over N_i}\sum_{m=1}^{i}\left(y_{m1}-\sum_{n=1}^{j} a_{mn}w_{n1}\right)^2$

**Gradient:**

$\displaystyle {d(loss) \over d(w_{j1})}=-{2 \over N_i}\sum_{m=1}^{i}\left[\left(a_{mj}\right)\left(y_{m1}-\sum_{n=1}^{j} a_{mn}w_{n1}\right)\right]$