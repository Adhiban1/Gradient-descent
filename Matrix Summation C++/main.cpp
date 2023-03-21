#include<iostream>
#include<array>
#include<cmath>
using namespace std;

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

double loss2(array<array<double, 1>, 3> Y, array<array<double, 1>, 3> Yhat){
    double l = 0;
    for(int i=0; i<Y.size(); i++){
        l += pow((Y[i][0] - Yhat[i][0]), 2);
    }
    l = l / Y.size();
    return l;
}

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

int main(){
    array<array<double, 4>, 3> A = {{{1,1,0,0},{0,0,0,0},{1,1,1,1}}};
    array<array<double, 1>, 4> W = {{{0},{0},{0},{0}}};
    array<array<double, 1>, 3> Y = {{{2},{0},{4}}};
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
}