import numpy as np
import matplotlib.pyplot as plt

np.random.seed(0)

X = np.random.randint(0, 101, (10, 1))
W = np.random.randint(0, 101, (1, 10))
B = np.random.randint(0, 101)
Y = W @ X + B

w = np.random.randint(0, 101, (1, 10))
b = np.random.randint(0, 101)

loss = ((Y - w @ X - b)**2).mean()

def dw(X, Y, w, b):
    return -2*X.T*(Y - w @ X - b)

def db(X, Y, w, b):
    return -2*(Y - w @ X - b)

lr = 1e-6
d = []
for i in range(100):
    w = w - lr*dw(X, Y, w, b)
    b = b - lr*db(X, Y, w, b)

    loss = ((Y - w @ X - b)**2).mean()

    d.append(loss)

print(W @ X + B, w @ X + b, sep='\n')

plt.plot(d)
plt.grid()
plt.ylabel('Loss')
plt.savefig('loss.png')
plt.close()