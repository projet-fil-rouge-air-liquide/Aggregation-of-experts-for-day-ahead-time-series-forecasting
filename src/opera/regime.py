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
    def __init__(self, n_outputs, lr=0.01, temp=4.0, eps=0.1, beta=0.01):
        self.n_outputs = n_outputs
        self.lr = lr
        self.temp = temp
        self.eps = eps
        self.beta = beta
        self.W = None

    def _init(self, n_features):
        self.W = 0.01 * np.random.randn(n_features, self.n_outputs)

    def predict(self, x):
        if self.W is None:
            self._init(len(x))
        x = x / (np.linalg.norm(x) + 1e-8) # normalize -> avoid vanishing
        logits = x @ self.W
        logits -= logits.max()
        exp = np.exp(logits / self.temp)
        p = exp / exp.sum()

        return self.eps / self.n_outputs + (1 - self.eps) * p

    def update(self, x, losses, p_target=None, target_strength=1.0):
        p = self.predict(x)

        losses = (losses - losses.mean()) / (losses.std() + 1e-8)
        baseline = np.dot(p, losses)

        grad = np.outer(x, losses - baseline)
        self.W -= self.lr * grad

        # entropy regularization
        entropy_grad = (p - 1/self.n_outputs)
        self.W -= self.lr * self.beta * np.outer(x, entropy_grad)

        if p_target is not None:
            p_grad = np.outer(x, (p - p_target))
            self.W -= self.lr * target_strength * p_grad

class TrendRegime(RegimePrior):
    def __init__(self, strength=0.1):
        self.strength = strength

    def bias(self, losses, features):
        ret = features[0] # WARNING: ret must be on the first col !
        direction = np.sign(ret)

        if direction > 0:
            losses[0] -= self.strength # bull 
            losses[1] += self.strength
        elif direction < 0:
            losses[0] += self.strength
            losses[1] -= self.strength

        return losses

class WindRegime(RegimePrior):
    def __init__(self, wind_feature_idx, wind_mean, wind_std, strength=0.5):
        self.idx = wind_feature_idx
        self.mean = wind_mean
        self.std = wind_std
        self.strength = strength

    def bias(self, losses, features):
        wind = features[self.idx]
        z = (wind - self.mean) / (self.std + 1e-8)
        s = np.tanh(z)  # in [-1, 1]

        # loss bias (still useful)
        losses[0] += self.strength * s   # low wind
        losses[1] -= self.strength * s   # high wind

        p_high = 0.5 * (1 + s)  # map [-1,1] → [0,1]
        p_target = np.array([1 - p_high, p_high])

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
        out = self.prior.bias(losses, features)

        if isinstance(out, tuple):
            losses, p_target = out
            self.gate.update(features, losses, p_target)
        else:
            self.gate.update(features, out)
