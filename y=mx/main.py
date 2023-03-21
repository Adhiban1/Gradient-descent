x = 2
m = 0
y = 4

f = lambda m, x : m * x
loss = lambda y, m, x : (y - f(m, x)) ** 2
dm = lambda y, m, x : -2 * x * (y - m * x)

lr = 0.1 # Leaarning Rate
for _ in range(10):
    m -= lr * dm(y, m, x)
    print(f'm: {m} | Loss: {loss(y, m, x)}')