#include<iostream>
#include<vector>
#include<cmath>
using namespace std;
typedef vector<vector<double>> Matrix;

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