#include "matrix.h"
#include <cmath>

Matrix dw(Matrix X, float Y, Matrix w, float b){
    return X.T()*(-2)*(Y - w.dot(X).mean() - b);
}

float db(Matrix X, float Y, Matrix w, float b){
    return (-2)*(Y - w.dot(X).mean() - b);
}

int main() {
    system("clear"); // Linux
    // system("cls") // Windows
    srand(1);
    
    Matrix X(10, 1), W(1, 10), w(1, 10);
    X.randfill(0, 100);
    W.randfill(0, 100);
    w.randfill(0, 100);
    
    float B = randNumber(0, 100), b = randNumber(0, 100);
    float Y = (W.dot(X) + B).mean();
    float loss = pow((Y - w.dot(X).mean() - b), 2);

    float lr = 1e-5;
    for(int i = 0; i < 55; i++){
        w = w - dw(X, Y, w, b)*lr;
        b = b - db(X, Y, w, b)*lr;
        float l = pow((Y - w.dot(X).mean() - b), 2);
        std::cout<<i<<". Loss: "<<l<<'\n';
    }
    std::cout<<"y true:      "<<W.dot(X).mean() + B<<'\n'<<"Predicted y: "<<w.dot(X).mean() + b<<'\n';
}

/*Output:
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
Predicted y: 10731.5*/