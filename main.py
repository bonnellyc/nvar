from viz.vizualisation_online_3D import live_plot_3d_trajectories
from scipy.integrate import solve_ivp

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
sol = solve_ivp(
    lorenz,
    t_span=(t_eval[0], t_eval[-1]),
    y0=[1.0, 1.0, 1.0],
    t_eval=t_eval,
)

trajectory = sol.y.T  # [n_timesteps, 3], convenient for 3D plotting
X = sol.y  # [3, n_timesteps], expected by NVAR

# MODEL
delay = 2
strides = [1, 3]
order = 2
update_rate = 0.5
alpha = 1.
tau=1

nvar = NVAR(delay=delay, strides= strides, order=order, update_rate=update_rate)
nvar.fit(X)
states = nvar.transform(X) # [valid_time_steps, n_features]


model = Ridge(alpha=alpha)
# on entraine le ridge sur les states à prédire les y
min_t = nvar.valid_index

X_data = states[:-tau, :]
y_data = X[:, min_t + tau:].T

split = int(0.7*len(X_data))

X_tr = X_data[split:]
y_tr = y_data[split:]

X_te = X_data[:split]
y_te = y_data[:split]

# ENTRAINEMENT DU MODELE
model.fit(X_tr, y_tr)

# 1 - on affiche la prediction du modele sur le X_data complet
y_pred_epoch = model.predict(X_data)

# 2 - on faiot tourner le modele en boucle fermée
# necessite quand meme les min_t 1er points
nvar.reset_state()

# warmup 
for t in range(min_t + 1):
        x_t = X[:, t]
        s_t = nvar.step(x_t)


# le nvar est CHAUD !
pred_online = []

# premier pred depuis s(t)
x_t = model.predict(s_t.reshape(1, -1))[0]
pred_online.append(x_t)

# boucle fermée
for t in range(min_t + tau + 1, X.shape[1]):
        s_t = nvar.step(x_t)

        x_t = model.predict(s_t.reshape(1, -1))[0]

        pred_online.append(x_t)

pred_online = np.asarray(pred_online)
print(pred_online.shape)

live_plot_3d_trajectories(
        trajectories=[pred_online],
        labels=["Pred"]
        )
