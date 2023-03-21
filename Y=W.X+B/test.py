from Matrix import Matrix, CMatrix
from time import time
import numpy as np
import pandas as pd

start = time()
for i in range(1000):
    a = Matrix([[1,2,3],[4,5,6],[7,8,9]])
    a = a.dot(a)
    a = 10*a
    a = a*a
    b = a.sum()
matrix_time = time()-start

start = time()
for i in range(1000):
    a = CMatrix([[1,2,3],[4,5,6],[7,8,9]])
    a = a.dot(a)
    a = 10*a
    a = a*a
    b = a.sum()
cmatrix_time = time()-start

start = time()
for i in range(1000):
    a = np.array([[1,2,3],[4,5,6],[7,8,9]])
    a = a.dot(a)
    a = 10*a
    a = a*a
    b = a.sum()
numpy_time = time()-start

df = pd.DataFrame({'Name':['Matrix', 'CMatrix', 'Numpy'], 'Time Taken':[matrix_time, cmatrix_time, numpy_time]})
df = df.sort_values('Time Taken')
df = df.reset_index(drop=True)
print(df)