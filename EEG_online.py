from river import linear_model, metrics, optim

import numpy as np
from viz.vizualisation_online_3D import live_plot_3d_trajectories
from scipy.integrate import solve_ivp
from pprint import pprint


from NVAR import NVAR

# recuperer les données EEG et les transformer en stream



def from_sample_to_dict(x_t, labels=None):
    if labels is None:
        labels = [f"x{c}" for c, _ in enumerate(x_t)]
    
    return {channel: float(val) for channel, val in zip(labels, x_t)}

delay = 2
strides = [1]
order = 2
update_rate = 1.0
alpha = 1.
tau=1


nvar = NVAR(delay=delay, strides= strides, order=order, update_rate=update_rate)

nvar.fit(X_scaled)
states = nvar.transform(X_scaled)

labels = nvar.feature_names(["x", "y", "z"])
min_t = nvar.valid_index

# River attend un dataset sous forme d'iterable de couples:
#   ({feature_name: feature_value, ...}, target)
#
# Ici on teste un modele scalaire simple: predire la coordonnee x de Lorenz
# a l'instant t + tau depuis les features NVAR a l'instant t.
X_data = states[:-tau, :]
y_data = X_scaled[:, min_t + tau:].T # [n_samples, n_dims]

output_names = ["x", "y", "z"]
dataset = [
    (from_sample_to_dict(x_t, labels), from_sample_to_dict(y_t, output_names))
    for x_t, y_t in zip(X_data, y_data)
]

models = {
    name: linear_model.LinearRegression(optimizer=optim.SGD(0.001))
    for name in output_names
}
scores = {name: metrics.R2() for name in output_names}

print_every = 200
y_preds_list = []

for step, (x_t, y_t) in enumerate(dataset, start=1):
    y_pred = {}
    
    for name in output_names:
        pred = models[name].predict_one(x_t)
        y_pred[name] = pred

        scores[name].update(y_true=y_t[name], y_pred=pred)
        models[name].learn_one(x_t, y_t[name])

    y_preds_list.append([y_pred[name] for name in output_names])

    if step % print_every == 0 or step == len(dataset):
        mean_r2 = np.mean([score.get() for score in scores.values()])
        print(
            f"[{step}] "
            f"R2_x={scores['x'].get():.6f} "
            f"R2_y={scores['y'].get():.6f} "
            f"R2_z={scores['z'].get():.6f} "
            f"mean={mean_r2:.6f}"
        )

preds_scaled = np.asarray(y_preds_list)  # [n_samples, 3]
preds = preds_scaled * X_std.ravel() + X_mean.ravel()
truth = y_data * X_std.ravel() + X_mean.ravel()

live_plot_3d_trajectories([truth, preds], labels=["truth", "river"])
