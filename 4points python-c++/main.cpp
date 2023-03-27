#include <iostream>
#include <array>
#include <cmath>

using namespace std;

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