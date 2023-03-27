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
