from flask import Flask, render_template, request
import matplotlib.pyplot as plt
import numpy as np
import imageio
import json

app = Flask(__name__)

@app.route('/', methods=['POST', 'GET'])
def home():
    l = ''
    with open('data.json') as f:
        data = json.load(f)
    if request.method == 'POST':
        n = int(request.form.get('iterations'))
        base = int(request.form.get('base'))
        power = int(request.form.get('power'))
        lr = float(request.form.get('lr'))
        points = request.form.get('points')
        fps = int(request.form.get('fps'))

        data = {'n':n, 'base':base, 'power':power, 'lr':lr, 'points':points}

        points = points.replace(' ', '')
        points = points.split('|')
        points = [list(map(float,i.split(','))) for i in points]

        A = np.array([
            [points[0][0]**3, points[0][0]**2, points[0][0], 1],
            [points[1][0]**3, points[1][0]**2, points[1][0], 1],
            [points[2][0]**3, points[2][0]**2, points[2][0], 1],
            [points[3][0]**3, points[3][0]**2, points[3][0], 1],
        ])

        W = np.zeros((len(points), 1))

        Y = np.array(points)[:, 1].reshape(-1, 1)


        def delta(A, W, Y, lr):
            return -lr*np.array([(2*A[:, 0].reshape(-1, 1)*(Y - A@W)).mean(), 
                                 (2*A[:, 1].reshape(-1, 1)*(Y - A@W)).mean(), 
                                 (2*A[:, 2].reshape(-1, 1)*(Y - A@W)).mean(), 
                                 (2*A[:, 3].reshape(-1, 1)*(Y - A@W)).mean()]).reshape(-1, 1)


        def loss(A, W, Y):
            return ((Y - A@W)**2).mean()


        def func(x, W):
            return (np.array([[x**3, x**2, x, 1]])@W)[0][0]


        x = np.linspace(np.array(points)[:, 0].min()-2,
                        np.array(points)[:, 0].max()+2, 100)
        y = [func(i, W) for i in x]
        x_y = [[i[0] for i in points], [i[1] for i in points]]

        writer = imageio.get_writer('static/graph.mp4', fps=fps)

        plt.ylim(Y.min()-2, Y.max()+2)

        plt.plot(x, y)
        plt.plot(*x_y, 'o')
        plt.savefig('image.jpg')
        plt.close()
        image = imageio.imread('image.jpg')
        writer.append_data(image)


        lr2 = np.linspace(1, np.power(base, 1/power), n)**power
        for i in range(n):
            last_l = loss(A, W, Y)
            W = W - lr2[i]*delta(A, W, Y, lr)
            l = loss(A, W, Y)
            if last_l < l:
                break
            print(i, l)
            y = [func(i, W) for i in x]

            plt.ylim(Y.min()-2, Y.max()+2)

            plt.plot(x, y)
            plt.plot(*x_y, 'o')
            plt.savefig('image.jpg')
            plt.close()
            image = imageio.imread('image.jpg')
            writer.append_data(image)
            if l < 1e-4:
                break
        writer.close()
        print(W)
    return render_template('index.html', loss=l, data=data)

if __name__ == '__main__':
    app.run(debug=True, port=5001)