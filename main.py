from viz.vizualisation_online_3D import live_plot_3d_trajectories
from scipy.integrate import solve_ivp
from pprint import pprint

import os
import numpy as np
from sklearn.linear_model import Ridge

from NVAR import NVAR

# DATA
def lorenz(t, state, sigma=10.0, rho=28.0, beta=8 / 3):
        x, y, z = state
        return [
            sigma * (y - x),
            x * (rho - z) - y,
            x * y - beta * z,
        ]

t_eval = np.linspace(0, 50, 1000)
dt = t_eval[1] - t_eval[0]
sol = solve_ivp(
    lorenz,
    t_span=(t_eval[0], t_eval[-1]),
    y0=[1.0, 1.0, 1.0],
    t_eval=t_eval,
)

trajectory = sol.y.T  # [n_timesteps, 3], convenient for 3D plotting
X = sol.y  # [3, n_timesteps], expected by NVAR

# Les features NVAR polynomiales explosent vite sur les coordonnées Lorenz
# brutes. On apprend et on déroule donc en coordonnées standardisées.
X_mean = X.mean(axis=1, keepdims=True)
X_std = X.std(axis=1, keepdims=True)
X_scaled = (X - X_mean) / X_std

# MODEL
delay = 2
strides = [1, 2, 3, 5]
order = 2
update_rate = 1.0
alpha = 1.
tau=1

nvar = NVAR(delay=delay, strides= strides, order=order, update_rate=update_rate)
nvar.fit(X_scaled)

states = nvar.transform(X_scaled) # [valid_time_steps, n_features]


model = Ridge(alpha=alpha)

min_t = nvar.valid_index

X_data = states[:-tau, :]
y_data = X_scaled[:, min_t+tau:].T

split = int(0.7*len(X_data))

X_tr = X_data[:split]
y_tr = y_data[:split]

X_te = X_data[split:]
y_te = y_data[split:]

# ENTRAINEMENT DU MODELE
model.fit(X_tr, y_tr)

# 1 - on affiche la prediction du modele sur le X_data complet
y_pred_epoch = model.predict(X_data)
y_pred_epoch = y_pred_epoch * X_std.ravel() + X_mean.ravel()

# 2 - on faiot tourner le modele en boucle fermée
# necessite quand meme les min_t 1er points
nvar.reset_state()

# warmup 
for t in range(min_t + 1):
        x_t = X_scaled[:, t]
        s_t = nvar.step(x_t)


# le nvar est CHAUD !
pred_online = []

# premier pred depuis s(t) : x(t+tau) = x(t) + dx
x_t = model.predict(s_t.reshape(1, -1))[0]
pred_online.append(x_t)

# boucle fermée
for t in range(min_t + tau + 1, X.shape[1]):
        s_t = nvar.step(x_t)

        x_t = model.predict(s_t.reshape(1, -1))[0]

        pred_online.append(x_t)

pred_online = np.asarray(pred_online)
pred_online = pred_online * X_std.ravel() + X_mean.ravel()

live_plot_3d_trajectories(trajectories=[y_pred_epoch, pred_online])
