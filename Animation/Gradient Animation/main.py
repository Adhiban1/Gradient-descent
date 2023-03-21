#!/home/adhiban/anaconda3/bin/python
import sympy
import numpy as np
import matplotlib.pyplot as plt
import imageio
import webbrowser
from rich.progress import track

def function(equation:str, xlim:tuple, x1:float, lr:float, epochs:int):
    x = sympy.symbols('x')
    f = sympy.sympify(equation)
    dx = f.diff(x)
    lf = sympy.lambdify(x, f, 'numpy')
    a = np.linspace(xlim[0], xlim[1], 100)
    x_data = [x1]
    y_data = [f.subs(x, x1)]
    writer = imageio.get_writer('graph.mp4')
    for _ in track(range(epochs)):
        x1 -= lr*dx.subs(x, x1)
        x_data.append(x1)
        y_data.append(f.subs(x, x1))
        plt.plot(a, lf(a))
        if len(x_data) >= 5:
            plt.plot(x_data[-5:], y_data[-5:], 'o-')
        else:
            plt.plot(x_data, y_data, 'o-')
        plt.savefig('graph.png')
        plt.close()
        image = imageio.imread('graph.png')
        writer.append_data(image)
    writer.close()
    plt.plot(a, lf(a))
    plt.plot(x_data, y_data, 'o-')
    plt.savefig('graph.png')
    plt.close()
    webbrowser.open('./index.html')

def main():
    equation = input('Equation: ')
    xlim_min = float(input('X - min: '))
    xlim_max = float(input('X - max: '))
    x1 = float(input('Initial value of x: '))
    lr = float(input('Learning rate: '))
    epochs = int(input('Epochs: '))
    function(equation, (xlim_min, xlim_max), x1, lr, epochs)

if __name__ == '__main__':
    main()