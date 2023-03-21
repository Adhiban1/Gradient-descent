#!/home/adhiban/anaconda3/bin/python
from flask import Flask, render_template, request
import sympy
import numpy as np
import matplotlib.pyplot as plt
import imageio
from rich.progress import track

def function(equation:str, xlim:tuple, x1:float, lr:float, epochs:int):
    x = sympy.symbols('x')
    f = sympy.sympify(equation)
    dx = f.diff(x)
    lf = sympy.lambdify(x, f, 'numpy')
    a = np.linspace(xlim[0], xlim[1], 100)
    x_data = [x1]
    y_data = [f.subs(x, x1)]
    writer = imageio.get_writer('./static/graph.mp4')
    for _ in track(range(epochs)):
        x1 -= lr*dx.subs(x, x1)
        x_data.append(x1)
        y_data.append(f.subs(x, x1))
        plt.plot(a, lf(a))
        if len(x_data) >= 5:
            plt.plot(x_data[-5:], y_data[-5:], 'o-')
        else:
            plt.plot(x_data, y_data, 'o-')
        plt.title(f'{equation}')
        plt.savefig('./static/image/graph.png')
        plt.close()
        image = imageio.imread('./static/image/graph.png')
        writer.append_data(image)
    plt.plot(a, lf(a))
    plt.plot(x_data, y_data, 'o-')
    plt.title(f'{equation}')
    plt.savefig('./static/image/graph.png')
    plt.close()
    writer.close()

app = Flask(__name__)

@app.route('/', methods=['POST', 'GET'])
def home():
    if request.method == 'POST':
        function(
            request.form.get('equation'),
            (float(request.form.get('x_min')), float(request.form.get('x_max'))),
            float(request.form.get('x1')),
            float(request.form.get('lr')),
            int(request.form.get('epochs'))
        )
    return render_template('index.html')


if __name__ == '__main__':
    app.run(debug=True)