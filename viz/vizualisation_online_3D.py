import numpy as np
import pyqtgraph as pg
from pyqtgraph.Qt import QtCore

try:
    import pyqtgraph.opengl as gl
except ModuleNotFoundError as exc:
    if exc.name == "OpenGL":
        raise ModuleNotFoundError(
            "pyqtgraph.opengl requires PyOpenGL. Install it with: poetry add PyOpenGL"
        ) from exc
    raise


DEFAULT_COLORS = [
    (1.0, 0.2, 0.2, 1.0),
    (0.2, 0.7, 1.0, 1.0),
    (0.2, 1.0, 0.45, 1.0),
    (1.0, 0.8, 0.2, 1.0),
    (0.8, 0.45, 1.0, 1.0),
    (1.0, 0.45, 0.75, 1.0),
]


def _as_points_3d(trajectory):
    points = np.asarray(trajectory, dtype=float)

    if points.ndim != 2:
        raise ValueError(f"expected a 2D trajectory, got shape {points.shape}")

    if points.shape[1] == 3:
        return points

    if points.shape[0] == 3:
        return points.T

    raise ValueError(
        "expected each trajectory to have shape (n_timesteps, 3) or "
        f"(3, n_timesteps), got {points.shape}"
    )


def live_plot_3d_trajectories(
    trajectories,
    labels=None,
    title="3D online trajectories",
    max_points=1000,
    interval_ms=20,
    colors=None,
    camera_distance=60,
    line_width=2,
):
    """
    Animate multiple 3D trajectories in the same OpenGL view.

    Parameters
    ----------
    trajectories:
        List of arrays. Each array must have shape (n_timesteps, 3) or
        (3, n_timesteps).
    labels:
        Optional list of model names. Used in the window title.
    max_points:
        Number of recent points kept visible for each trajectory.
    interval_ms:
        Timer interval in milliseconds.
    """
    if len(trajectories) == 0:
        raise ValueError("trajectories must contain at least one trajectory")

    points_by_model = [_as_points_3d(trajectory) for trajectory in trajectories]
    n_models = len(points_by_model)

    if labels is None:
        labels = [f"model_{i}" for i in range(n_models)]
    elif len(labels) != n_models:
        raise ValueError(f"expected {n_models} labels, got {len(labels)}")

    colors = colors or DEFAULT_COLORS

    app = pg.mkQApp()
    view = gl.GLViewWidget()
    view.setWindowTitle(f"{title} | " + " | ".join(labels))
    view.setCameraPosition(distance=camera_distance)

    grid = gl.GLGridItem()
    grid.scale(5, 5, 1)
    view.addItem(grid)

    curves = []
    buffers = [[] for _ in range(n_models)]

    for i in range(n_models):
        curve = gl.GLLinePlotItem(
            pos=np.empty((0, 3)),
            color=colors[i % len(colors)],
            width=line_width,
            antialias=True,
        )
        view.addItem(curve)
        curves.append(curve)

    step = {"index": 0}
    max_timesteps = max(points.shape[0] for points in points_by_model)

    def update():
        t = step["index"]

        if t >= max_timesteps:
            timer.stop()
            return

        for i, points in enumerate(points_by_model):
            if t >= points.shape[0]:
                continue

            buffers[i].append(points[t])
            buffers[i] = buffers[i][-max_points:]
            curves[i].setData(pos=np.asarray(buffers[i]))

        step["index"] += 1

    timer = QtCore.QTimer()
    timer.timeout.connect(update)
    timer.start(interval_ms)

    view.show()
    app.exec()


if __name__ == "__main__":
    from scipy.integrate import solve_ivp

    def lorenz(t, state, sigma=10.0, rho=28.0, beta=8 / 3):
        x, y, z = state
        return [
            sigma * (y - x),
            x * (rho - z) - y,
            x * y - beta * z,
        ]

    t_eval = np.linspace(0, 50, 10000)
    sol = solve_ivp(
        lorenz,
        t_span=(t_eval[0], t_eval[-1]),
        y0=[1.0, 1.0, 1.0],
        t_eval=t_eval,
    )

    y_true = sol.y.T
    y_shifted = y_true + np.array([2.0, 0.0, 0.0])

    live_plot_3d_trajectories(
        [y_true, y_shifted],
        labels=["true", "shifted"],
        max_points=1500,
    )
