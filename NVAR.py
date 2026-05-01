import numpy as np
from itertools import combinations_with_replacement
from math import comb

class NVAR:
    def __init__(self, delay:int, strides:int | list, order:int, update_rate:float):
        self.delay = delay
        self.strides = [strides] if isinstance(strides, int) else strides
        self.order = order
        self.update_rate = update_rate
        self.is_initialised = False
        self.S = None
        self._history = []
        self._poly_indices = []
    
    @property
    def n_features(self):
        if not self.is_initialised:
            raise AttributeError("n_features is unknown before fit(X).")

        return sum(
            comb(self.n_linear_features + degree - 1, degree)
            for degree in range(1, self.order + 1)
        )
    
    @property
    def n_linear_features(self):
        if not self.is_initialised:
            raise AttributeError("n_linear_features is unknown before fit(X).")

        return self.delay * len(self.strides) * self.n_dims_input

    def _linear_features_at(self, X:np.ndarray, t:int):
        """
        X : [n_dims, n_timesteps]

        return : z(t) = [x(t), x(t-tau), ..., x(t-delay*tau)]
        """
        features = []
        # creer le vecteur composé de self.delay termes, séparé par self.strides timesteps
        for d in range(self.delay):
            for stride in self.strides:
                lag = d * stride
                features.append(X[:, t-lag])
        
        return np.concatenate(features)
    
    def _format_lagged_name(self, channel_name: str, lag: int):
        if lag == 0:
            return f"{channel_name}(t)"

        return f"{channel_name}(t-{lag})"
    
    def linear_feature_names(self, channel_names=None):
        """
        Return the names of the linear features, in the same order as
        _linear_features_at.
        """
        if not self.is_initialised:
            raise AttributeError("feature names are unknown before fit(X) or step(x_t).")

        if channel_names is None:
            channel_names = [f"x{i}" for i in range(self.n_dims_input)]
        elif len(channel_names) != self.n_dims_input:
            raise ValueError(
                f"expected {self.n_dims_input} channel names, got {len(channel_names)}"
            )

        names = []
        for d in range(self.delay):
            for stride in self.strides:
                lag = d * stride
                for channel_name in channel_names:
                    names.append(self._format_lagged_name(channel_name, lag))

        return names
    
    def feature_names(self, channel_names=None):
        """
        Return all feature names, linear first, then polynomial features of
        degree 2..order, in the same order as transform/step outputs.
        """
        linear_names = self.linear_feature_names(channel_names)
        names = list(linear_names)

        for degree_indices in self._poly_indices:
            for indices in degree_indices:
                factors = [linear_names[index] for index in indices]
                counts = {}
                for factor in factors:
                    counts[factor] = counts.get(factor, 0) + 1

                parts = [
                    factor if power == 1 else f"{factor}^{power}"
                    for factor, power in counts.items()
                ]
                names.append("*".join(parts))

        return names
    
    @property
    def _max_lag(self):
        return (self.delay - 1) * np.max(self.strides)
    
    @property
    def valid_index(self):
        return (self.delay - 1) * max(self.strides)

    
    def _build_poly_indices(self):
        self._poly_indices = [
            tuple(combinations_with_replacement(range(self.n_linear_features), degree))
            for degree in range(2, self.order + 1)
        ]
    
    def _non_linear_features(self, linear_features: np.ndarray):
        """
        linear_features : [n_linear_features]

        return : polynomial features of degree 2..order, without duplicates.
        """
        features = []

        for degree_indices in self._poly_indices:
            for indices in degree_indices:
                features.append(np.prod(linear_features[list(indices)]))

        return np.asarray(features)
    
    def _features_from_linear(self, linear_features: np.ndarray):
        if self.order == 1:
            return linear_features

        non_linear_features = self._non_linear_features(linear_features)
        return np.concatenate([linear_features, non_linear_features])
    
    def _update_state(self, z_t):
        if self.S is None:
            self.S = z_t
        
        else:
            alpha = self.update_rate
            self.S = (1 - alpha)*self.S + alpha*z_t
        
        return self.S
    
    def reset_state(self):
        self.S = None
        self._history = []
    

    def fit(self, X, y=None):
        
        if self.order < 1:
            raise ValueError("order must be greater than or equal to 1")

        # initialise le NVAR
        self._validate(X)
        self.n_dims_input = X.shape[0]
        self.is_initialised = True
        self._build_poly_indices()
        
        return self
    
    def transform(self, X, y=None): 
        """
        X : [n_dims, n_timesteps]
        """
        # renvoie une matrice de tout les etats s(t) concaténés
        self._validate(X)
        self.S = None

        min_t = self._max_lag
        states = []

        for t in range(min_t, X.shape[1]):
            linear_features = self._linear_features_at(X, t)
            z_t = self._features_from_linear(linear_features)
            s_t = self._update_state(z_t)
            states.append(s_t.copy())

        return np.stack(states, axis=0)

    def step(self, x_t):
        """
        x_t = [n_dims,]
        return: S(t) = [n_features,], or None while history is too short.
        """
        x_t = np.asarray(x_t)
        if x_t.ndim != 1:
            raise IndexError(f"mauvaise shape : expected [n_dims] : got {x_t.shape}")
        
        if not self.is_initialised:
            self.n_dims_input = x_t.shape[0]
            self.is_initialised = True
            self._build_poly_indices()
        elif x_t.shape[0] != self.n_dims_input:
            raise IndexError(f"mauvaise shape : expected [{self.n_dims_input}] : got {x_t.shape}")

        # on ajoute le point dans le buffer interne _history
        self._history.append(x_t)
        # si pas suffisament de point dans le buffer
        if len(self._history) <= self._max_lag:
            return None

        linear_features = []
        for d in range(self.delay):
            for stride in self.strides:
                lag = d * stride
                linear_features.append(self._history[-1 - lag])
        
        linear_features = np.concatenate(linear_features)
        z_t = self._features_from_linear(linear_features)
        return self._update_state(z_t)
    
    def _validate(self, X):
        # s'assure que X est dans le bon format
        if X.ndim != 2:
            raise IndexError(f"mauvaise shape : expected [n_dims, n_timesteps] : got {X.shape}")

        


if __name__ == "__main__":
    import matplotlib.pyplot as plt
    from pprint import pprint

    delay = 2
    strides = [1, 3]
    order = 2
    update_rate = 1.

    nvar = NVAR(delay, strides, order, update_rate)

    t = np.linspace(0, 100, 1000)
    X_1 = np.sin(t)
    X_2 = np.cos(t)
    X = np.stack([X_1, X_2], axis=0)
    print(X.shape)

    nvar.fit(X)
    pprint(nvar.feature_names(['Cz', 'Fp1']), width=120)

    states = nvar.transform(X) # [valide_timestep, n_features]
    #print(states)
