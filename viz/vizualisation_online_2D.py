import numpy as np
import pyqtgraph as pg
from pyqtgraph.Qt import QtCore
from NVAR import NVAR

app = pg.mkQApp()

win = pg.GraphicsLayoutWidget(title="NVAR live output")
plot = win.addPlot(title="s(t)")

delay = 3
strides = [1, 3]
order = 1
update_rate = 1.

t = np.linspace(0, 100, 1000)
X_1 = np.tanh(t%np.pi/2)
X_2 = np.cos(t)
X = np.stack([X_1, X_2], axis=0)

nvar = NVAR(delay, strides, order, update_rate)
nvar.fit(X)

curves = [plot.plot(pen=i) for i in range(nvar.n_features)]
buffers = [[] for i in range(nvar.n_features)]


def get_next_sample(X):
    for t in range(X.shape[1]):
        yield X[:, t]

stream = get_next_sample(X)

def update():
    global buffers

    try:
        x_t = next(stream)          # shape: (n_dims,)
    except StopIteration:
        timer.stop()
        return

    s_t = nvar.step(x_t)

    if s_t is not None:
        for i in range(nvar.n_features):
            buffers[i].append(s_t[i])
            buffers[i] = buffers[i][-500:]     # garde les 500 derniers points
            curves[i].setData(buffers[i])

timer = QtCore.QTimer()
timer.timeout.connect(update)
timer.start(20)  # update toutes les 20 ms

win.show()
app.exec()
