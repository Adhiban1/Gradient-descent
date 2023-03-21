#include "matrix.h"
#include<unistd.h>
#include <chrono>

using namespace std::chrono;

int main() {
    auto start = high_resolution_clock::now();
    for(int i = 0; i < 1000; i++){
        Matrix a(3,3);
        a.matrix = {{1,2,3},{4,5,6},{7,8,9}};
        a = a.dot(a);
        a = a*10;
        a = a*a;
        float b = a.sum();
    }
    auto stop = high_resolution_clock::now();
    auto CPPmatrix_time = duration_cast<microseconds>(stop-start);
    std::cout<<CPPmatrix_time.count()/(1e+6)<<'\n';
}

/*
output:
0.047824
*/