import math

import numpy as np


class SGD:
    def __init__(self, learning_rate=0.01):
        self.learning_rate = learning_rate

    def update(self, weights, bias, weight_gradients, bias_gradient):
        updated_weights = [
            weight - self.learning_rate * gradient
            for weight, gradient in zip(weights, weight_gradients)
        ]

        updated_bias = bias - self.learning_rate * bias_gradient

        return updated_weights, updated_bias


class Adam:
    """
    Adam proprietario con decoupled weight decay opzionale.

    Espone `state_dict()` / `load_state_dict()` per permettere il **resume**
    del training senza perdere i momenti accumulati.
    """

    def __init__(self, learning_rate=0.001, beta1=0.9, beta2=0.999, epsilon=1e-8,
                 weight_decay=0.0):
        self.learning_rate = learning_rate
        self.beta1 = beta1
        self.beta2 = beta2
        self.epsilon = epsilon
        self.weight_decay = float(weight_decay)
        self.t = 0
        self.m = None
        self.v = None

    def set_learning_rate(self, learning_rate):
        """Aggiorna il learning rate (usato dallo scheduler)."""
        if learning_rate < 0:
            raise ValueError("learning_rate non può essere negativo")
        self.learning_rate = float(learning_rate)
        return self.learning_rate

    def update(self, parameters, gradients):
        parameters = list(parameters)
        gradients = list(gradients)

        if len(parameters) != len(gradients):
            raise ValueError("parameters e gradients devono avere la stessa lunghezza.")

        if self.m is None:
            self.m = [np.zeros_like(p) for p in parameters]
            self.v = [np.zeros_like(p) for p in parameters]

        self.t += 1

        updated = []
        for i, (param, grad) in enumerate(zip(parameters, gradients)):
            self.m[i] = self.beta1 * self.m[i] + (1 - self.beta1) * grad
            self.v[i] = self.beta2 * self.v[i] + (1 - self.beta2) * (grad ** 2)

            m_hat = self.m[i] / (1 - self.beta1 ** self.t)
            v_hat = self.v[i] / (1 - self.beta2 ** self.t)

            param -= self.learning_rate * m_hat / (np.sqrt(v_hat) + self.epsilon)

            if self.weight_decay > 0 and np.ndim(param) > 1:
                param -= self.learning_rate * self.weight_decay * param

            updated.append(param)

        return updated

    def reset(self):
        """Azzera i momenti accumulati."""
        self.t = 0
        self.m = None
        self.v = None

    def state_dict(self):
        return {
            "learning_rate": self.learning_rate,
            "beta1": self.beta1,
            "beta2": self.beta2,
            "epsilon": self.epsilon,
            "weight_decay": self.weight_decay,
            "t": self.t,
            "m": None if self.m is None else [np.asarray(x).copy() for x in self.m],
            "v": None if self.v is None else [np.asarray(x).copy() for x in self.v],
        }

    def load_state_dict(self, state):
        if not isinstance(state, dict):
            raise ValueError("Stato dell'ottimizzatore non valido.")

        self.learning_rate = state.get("learning_rate", self.learning_rate)
        self.beta1 = state.get("beta1", self.beta1)
        self.beta2 = state.get("beta2", self.beta2)
        self.epsilon = state.get("epsilon", self.epsilon)
        self.weight_decay = state.get("weight_decay", self.weight_decay)
        self.t = int(state.get("t", 0))

        m = state.get("m")
        v = state.get("v")
        self.m = None if m is None else [np.asarray(x, dtype=float) for x in m]
        self.v = None if v is None else [np.asarray(x, dtype=float) for x in v]
        return self


class LearningRateScheduler:
    """
    Scheduler di learning rate con **warmup** lineare e decadimento
    configurabile.

    Schedule supportate:
        - ``"constant"``: learning rate costante dopo il warmup.
        - ``"linear"``:   decadimento lineare fino a `min_learning_rate`.
        - ``"cosine"``:   decadimento a coseno fino a `min_learning_rate`.

    Il warmup evita la divergenza iniziale tipica dei Transformer.
    """

    SUPPORTED_SCHEDULES = ("constant", "linear", "cosine")

    def __init__(self, base_learning_rate, warmup_steps=0, total_steps=None,
                 schedule="cosine", min_learning_rate=0.0):
        if base_learning_rate <= 0:
            raise ValueError("base_learning_rate deve essere maggiore di 0")
        if warmup_steps < 0:
            raise ValueError("warmup_steps non può essere negativo")
        if total_steps is not None and total_steps <= 0:
            raise ValueError("total_steps deve essere maggiore di 0")
        if schedule not in self.SUPPORTED_SCHEDULES:
            raise ValueError(
                f"schedule deve essere uno di {self.SUPPORTED_SCHEDULES}"
            )
        if min_learning_rate < 0 or min_learning_rate > base_learning_rate:
            raise ValueError(
                "min_learning_rate deve essere tra 0 e base_learning_rate"
            )

        self.base_learning_rate = float(base_learning_rate)
        self.warmup_steps = int(warmup_steps)
        self.total_steps = None if total_steps is None else int(total_steps)
        self.schedule = schedule
        self.min_learning_rate = float(min_learning_rate)
        self.last_step = 0

    def learning_rate(self, step):
        """
        Learning rate per lo step indicato (0-based: `step=0` è il primo update).
        """
        step = int(step)
        if step < 0:
            raise ValueError("step non può essere negativo")

        if self.warmup_steps > 0 and step < self.warmup_steps:
            return self.base_learning_rate * (step + 1) / self.warmup_steps

        if self.schedule == "constant" or self.total_steps is None:
            return self.base_learning_rate

        decay_steps = max(1, self.total_steps - self.warmup_steps)
        progress = (step - self.warmup_steps) / decay_steps
        progress = min(1.0, max(0.0, progress))

        span = self.base_learning_rate - self.min_learning_rate

        if self.schedule == "linear":
            return self.base_learning_rate - span * progress

        return self.min_learning_rate + 0.5 * span * (1.0 + math.cos(math.pi * progress))

    def step(self):
        """Avanza di uno step e restituisce il learning rate corrente."""
        value = self.learning_rate(self.last_step)
        self.last_step += 1
        return value

    def state_dict(self):
        return {
            "base_learning_rate": self.base_learning_rate,
            "warmup_steps": self.warmup_steps,
            "total_steps": self.total_steps,
            "schedule": self.schedule,
            "min_learning_rate": self.min_learning_rate,
            "last_step": self.last_step,
        }

    def load_state_dict(self, state):
        if not isinstance(state, dict):
            raise ValueError("Stato dello scheduler non valido.")
        self.base_learning_rate = float(state.get("base_learning_rate", self.base_learning_rate))
        self.warmup_steps = int(state.get("warmup_steps", self.warmup_steps))
        total_steps = state.get("total_steps", self.total_steps)
        self.total_steps = None if total_steps is None else int(total_steps)
        self.schedule = state.get("schedule", self.schedule)
        self.min_learning_rate = float(state.get("min_learning_rate", self.min_learning_rate))
        self.last_step = int(state.get("last_step", 0))
        return self
