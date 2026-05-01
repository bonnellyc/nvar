from river import linear_model, metrics, optim

import numpy as np
from viz.vizualisation_online_3D import live_plot_3d_trajectories
from scipy.integrate import solve_ivp
from pprint import pprint


from NVAR import NVAR

def lorenz(t, state, sigma=10.0, rho=28.0, beta=8 / 3):
        x, y, z = state
        return [
            sigma * (y - x),
            x * (rho - z) - y,
            x * y - beta * z,
        ]

t_eval = np.linspace(0, 50, 10000)
dt = t_eval[1] - t_eval[0]
sol = solve_ivp(
    lorenz,
    t_span=(t_eval[0], t_eval[-1]),
    y0=[1.0, 1.0, 1.0],
    t_eval=t_eval,
)

trajectory = sol.y.T  # [n_timesteps, 3], convenient for 3D plotting
X = sol.y  # [3, n_timesteps], expected by NVAR

X_mean = X.mean(axis=1, keepdims=True)
X_std = X.std(axis=1, keepdims=True)
X_scaled = (X - X_mean) / X_std


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

split = int(0.7 * X_scaled.shape[1])

X_scaled_train = X_scaled[:, :split]
y_scaled_train = X_scaled[:, :split]

X_scaled_test = X_scaled[:,split:]
y_scaled_test = X_scaled[:,split:]

nvar.fit(X_scaled_train)
states_train = nvar.transform(X_scaled_train)

labels = nvar.feature_names(["x", "y", "z"])
min_t = nvar.valid_index

X_train = states_train[:-tau, :]
x_train_current = X_scaled_train[:, min_t:-tau].T
y_train = X_scaled_train[:, min_t + tau:].T
dy_train = y_train - x_train_current

# River attend un dataset sous forme d'iterable de couples:
#   ({feature_name: feature_value, ...}, target)
#
# Ici on teste un modele scalaire simple: predire la coordonnee x de Lorenz
# a l'instant t + tau depuis les features NVAR a l'instant t.
 # [n_samples, n_dims]

# ON VA S'ENTRAINER SUR LE TRAIN PUIS LAISSER LE MODELE TOURNER TOUT SEUL SUR LE TEST

output_names = ["x", "y", "z"]
delta_names = ["dx", "dy", "dz"]
dataset_train = [
    (from_sample_to_dict(x_t, labels), from_sample_to_dict(dy_t, delta_names))
    for x_t, dy_t in zip(X_train, dy_train)
]

models = {
    name: linear_model.LinearRegression(optimizer=optim.SGD(0.0001))
    for name in delta_names
}
delta_scores = {name: metrics.R2() for name in delta_names}
position_scores = {name: metrics.R2() for name in output_names}

print_every = 200

train_preds_scaled = []

for step, (x_t, y_t) in enumerate(dataset_train, start=1):
    y_pred_delta = {}
    
    for name in delta_names:
        pred = models[name].predict_one(x_t)
        y_pred_delta[name] = pred

        delta_scores[name].update(y_true=y_t[name], y_pred=pred)
        models[name].learn_one(x_t, y_t[name])

    delta_pred = np.array([y_pred_delta[name] for name in delta_names])
    position_pred = x_train_current[step - 1] + delta_pred
    train_preds_scaled.append(position_pred)

    for i, name in enumerate(output_names):
        position_scores[name].update(y_true=y_train[step - 1, i], y_pred=position_pred[i])

    if step % print_every == 0 or step == len(dataset_train):
        mean_delta_r2 = np.mean([score.get() for score in delta_scores.values()])
        mean_position_r2 = np.mean([score.get() for score in position_scores.values()])
        print(
            f"[{step}] "
            f"R2_pos_x={position_scores['x'].get():.6f} "
            f"R2_pos_y={position_scores['y'].get():.6f} "
            f"R2_pos_z={position_scores['z'].get():.6f} "
            f"mean_pos={mean_position_r2:.6f} "
            f"mean_delta={mean_delta_r2:.6f}",
            flush=True,
        )


def predict_delta(s_t):
    s_dict = from_sample_to_dict(s_t, labels)
    return np.array([
        models["dx"].predict_one(s_dict),
        models["dy"].predict_one(s_dict),
        models["dz"].predict_one(s_dict),
    ])

# PUIS LAISSE TOURNER SUR LE TEST
nvar.reset_state()

# warmup 
for t in range(min_t + 1):
    x_t = X_scaled_test[:, t]
    s_t = nvar.step(x_t)


# premier pred depuis s(t)
x_t = X_scaled_test[:, min_t] + predict_delta(s_t)


# boucle fermée
pred_online = [x_t]

for _ in range(min_t + tau + 1, X_scaled_test.shape[1]):
    # on crée le vecteur de feature à partir de x_t scaled
    s_t = nvar.step(x_t)
    x_t = x_t + predict_delta(s_t)

    pred_online.append(x_t)

preds_scaled = np.asarray(pred_online)  # [n_samples, 3]
truth_test_scaled = X_scaled_test[:, min_t + tau:].T

preds = preds_scaled * X_std.ravel() + X_mean.ravel()
truth_test = truth_test_scaled * X_std.ravel() + X_mean.ravel()


train_preds = np.asarray(train_preds_scaled) * X_std.ravel() + X_mean.ravel()
truth_train = y_train * X_std.ravel() + X_mean.ravel()
total_preds = np.concatenate([train_preds, preds], axis=0)
total_truth = np.concatenate([truth_train, truth_test], axis=0)

print(f"train samples: {len(dataset_train)}", flush=True)
print(f"test rollout samples: {len(preds)}", flush=True)

live_plot_3d_trajectories([total_truth, total_preds], labels=["truth", "river"])
