import ctypes
import numpy as np
import plotly.express as px
import plotly.graph_objs as go
from flask import Flask, render_template, request
import re

def func(x, w):
    return w[0]*x**3+w[1]*x**2+w[2]*x+w[3]

clib = ctypes.CDLL('./grad.so')
clib.function.argtypes = [ctypes.POINTER(ctypes.c_double), 
                          ctypes.POINTER(ctypes.c_double), 
                          ctypes.c_int, ctypes.c_double]
clib.function.restype = ctypes.POINTER(ctypes.c_double)

app = Flask(__name__)

@app.route('/', methods=['POST', 'GET'])
def home():
    fig = px.line(x=[0], y=[0])
    loss = 'NaN'
    equ = ''
    if request.method == 'POST':
        w = (ctypes.c_double*5)(0,0,0,0,0)
        # arr = [-1, 1, 0, 0, 1, 1, 2, 0]
        arr = list(map(float, re.findall('[-\d\.]+', request.form.get('points'))))

        x1 = [arr[i*2] for i in range(4)]
        y1 = [arr[i*2+1] for i in range(4)]

        clib.function(
            (ctypes.c_double*8)(*arr),
            w,
            1000,
            0.01
        )

        loss = w[4]

        x = np.linspace(min(x1)-1, max(x1)+1, 100)
        y = func(x, w)

        # plt.plot(x, y)
        # plt.plot(x1, y1, 'o', color='red')
        # plt.title(f'${w[0]:.2f}x^3+{w[1]:.2f}x^2+{w[2]:.2f}x+{w[3]:.2f}$')
        # plt.show()
        fig1 = px.line(x=x, y=y)
        fig2 = px.scatter(x=x1, y=y1)
        fig = go.Figure(data=[fig1.data[0], fig2.data[0]])
        fig.data[1].marker.color = '#ff0000'
        fig.update_layout(
            xaxis_title="X Axis",
            yaxis_title="Y Axis"
        )
        equ = f'\\({w[0]:.2f}x^3+{w[1]:.2f}x^2+{w[2]:.2f}x+{w[3]:.2f}\\)'
    return render_template('index.html', graph=fig.to_json(), loss=loss, 
                           equ=equ)

if __name__ == '__main__':
    app.run(debug=True)