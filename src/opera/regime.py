import pandas as pd
import numpy as np

class RegimePrior:
    """
    Base class for regime priors.
    A prior can bias regime losses before the gate update.
    """
    def bias(self, losses, features):
        return losses

class SoftmaxGate:
    def __init__(self, n_outputs, lr=0.01, temp=2.0, eps=0.1, beta=0.01):
        self.n_outputs = n_outputs
        self.lr = lr
        self.temp = temp
        self.eps = eps
        self.beta = beta
        self.W = None

    def _init(self, n_features):
        scale = 1.0 / np.sqrt(n_features)
        self.W = scale * np.random.randn(n_features, self.n_outputs).astype(np.float32)

    def _forward(self, x):
        x = x.astype(np.float32)
        x = x / (np.linalg.norm(x) + 1e-8)

        logits = x @ self.W
        logits -= logits.max()

        exp = np.exp(logits / self.temp)
        p = exp / (exp.sum() + 1e-8)

        p = self.eps / self.n_outputs + (1.0 - self.eps) * p
        return x, p

    def predict(self, x):
        if self.W is None:
            self._init(len(x))
        _, p = self._forward(x)
        return p

    def update(self, x, losses, p_target=None, target_strength=1.0):
        if self.W is None:
            self._init(len(x))

        x, p = self._forward(x)

        # robust loss normalization
        losses = losses - losses.mean()
        losses /= (np.mean(np.abs(losses)) + 1e-8)
        losses = np.clip(losses, -3.0, 3.0)

        baseline = np.dot(p, losses)
        delta = losses - baseline

        # policy gradient
        self.W -= self.lr * x[:, None] * delta[None, :]

        # entropy regularization (stable)
        entropy_grad = p * (np.log(p + 1e-8) + 1.0)
        self.W -= self.lr * self.beta * x[:, None] * entropy_grad[None, :]

        # optional target distribution
        if p_target is not None:
            self.W -= self.lr * target_strength * x[:, None] * (p - p_target)[None, :]

class TrendRegime(RegimePrior):
    def __init__(self, trend_idx, strength=0.2, inertia=0.8):
        self.idx = trend_idx
        self.strength = float(strength)
        self.inertia = float(inertia)
        self.prev_probs = np.array([0.5, 0.5])

    def _bull_bear_strengths(self, features):
        z = float(features[self.idx])
        bull = np.tanh(np.maximum(z, 0.0))
        bear = np.tanh(np.maximum(-z, 0.0))
        return bull, bear

    def bias(self, losses, features):
        losses = losses.copy()

        bull, bear = self._bull_bear_strengths(features)

        raw = np.array([bull, bear])
        raw /= raw.sum() + 1e-8

        self.prev_probs += (1.0 - self.inertia) * (raw - self.prev_probs)

        directional = bull - bear
        losses[0] -= self.strength * directional
        losses[1] += self.strength * directional

        return losses, self.prev_probs

class WindRegime(RegimePrior):
    def __init__(self, wind_feature_idx, wind_mean, wind_std, strength=0.5):
        self.idx = wind_feature_idx
        self.mean = float(wind_mean)
        self.std = float(wind_std)
        self.strength = float(strength)

    def bias(self, losses, features):
        losses = losses.copy()

        wind = float(features[self.idx])
        z = (wind - self.mean) / (self.std + 1e-8)
        z = np.clip(z, -3.0, 3.0)

        s = np.tanh(z)
        confidence = abs(s)

        if confidence <= 0.1:
            return losses, np.array([0.5, 0.5])

        direction = 1.0 if s > 0 else -1.0
        bias = self.strength * confidence

        # index 0 = low wind, index 1 = high wind
        losses[0] += direction * bias
        losses[1] -= direction * bias

        shift = 0.5 * confidence
        p_high = np.clip(0.5 + direction * shift, 0.0, 1.0)
        p_target = np.array([1.0 - p_high, p_high], dtype=np.float32)

        return losses, p_target

class Regime:
    """
    represents one latent dimension of the environment
    (e.g. trend, wind, volatility).

    It wraps a gate and exposes a uniform interface to HMoE.
    """
    def __init__(self, name, regimes, gate, prior=None):
        """
        name    : str, name of the axis ("trend", "wind", ...)
        regimes : list[str], regime labels (["bull", "bear"])
        gate    : SoftmaxGate
        prior   : optional RegimePrior
        """
        self.name = name
        self.regimes = regimes
        self.gate = gate
        self.prior = prior if prior is not None else RegimePrior()

    def predict(self, features):
        """Return P(regime | features)."""
        return self.gate.predict(features)

    def update(self, features, losses):
        losses, p_target = self.prior.bias(losses, features)
        self.gate.update(features, losses, p_target)
