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
    def __init__(self, n_outputs, lr=0.01, temp=1.0, eps=0.1, beta=0.0):
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

class UpDownRegime(RegimePrior):
    """
    Trend Regime Up or Down (short term)
    """
    def __init__(self, trend_idx, strength=0.2, inertia=0.8):
        self.idx = trend_idx
        self.strength = float(strength)
        self.inertia = float(inertia)
        self.prev_probs = np.array([0.5, 0.5])

    def _up_down_strengths(self, features):
        z = float(features[self.idx])
        up = np.tanh(np.maximum(z, 0.0))
        down = np.tanh(np.maximum(-z, 0.0))
        return up, down

    def bias(self, losses, features):
        losses = losses.copy()

        up, down = self._up_down_strengths(features)

        raw = np.array([up, down])
        raw /= raw.sum() + 1e-8

        self.prev_probs += (1.0 - self.inertia) * (raw - self.prev_probs)

        directional = up - down
        losses[0] -= self.strength * directional
        losses[1] += self.strength * directional

        return losses, self.prev_probs

class BullBearRegime(RegimePrior):
    """
    Trend Regime Bull or Bear
    """
    def __init__(self, strength=0.2, inertia=0.95):
        self.strength = float(strength)
        self.inertia = float(inertia)

        self.prev_probs = np.array([0.5, 0.5])

        # long term memory
        self.smooth_score = 0.0
        self.alpha = 0.05

        # thresholds
        self.bull_th = 0.2
        self.bear_th = -0.2

    def _trend_score(self, features):
        trend_strength, mom_24, mom_48, vol = features

        score = (
            0.5 * trend_strength +
            0.3 * mom_48 +
            0.2 * mom_24
        )

        # penality for high volatility (uncertainty)
        score /= (1.0 + vol)

        return score
        
    def _bull_bear_strengths(self, features):
        score = self._trend_score(features)

        self.smooth_score = (
            (1 - self.alpha) * self.smooth_score +
            self.alpha * score
        )

        s = self.smooth_score

        if s > self.bull_th:
            bull, bear = 1.0, 0.0
        elif s < self.bear_th:
            bull, bear = 0.0, 1.0
        else:
            bull, bear = 0.5, 0.5

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

class VolatilityRegime(RegimePrior):
    def __init__(self, vol_idx, low_th, high_th, strength=0.3):
        self.idx = vol_idx
        self.low_th = low_th
        self.high_th = high_th
        self.strength = strength

    def bias(self, losses, features):
        losses = losses.copy()
        vol = float(features[self.idx])

        if vol > self.high_th:
            # regime high vol
            losses[0] += self.strength
            losses[1] -= self.strength
            p = np.array([0.0, 1.0])
        elif vol < self.low_th:
            # regime low vol
            losses[0] -= self.strength
            losses[1] += self.strength
            p = np.array([1.0, 0.0])
        else:
            p = np.array([0.5, 0.5])

        return losses, p
############################
# -- Determinist Regime -- #
############################

class DayNightRegime:
    def __init__(self, hour_idx=0, day_start=6, day_end=21):
        self.idx = hour_idx
        self.day_start = day_start
        self.day_end = day_end

    def predict(self, features):
        hour = float(features[self.idx])
        is_day = (hour >= self.day_start) and (hour < self.day_end)

        if is_day:
            return np.array([1.0, 0.0])  # day
        else:
            return np.array([0.0, 1.0])  # night

    def bias(self, losses, features):
        losses = losses.copy()
        hour = float(features[self.idx])
        is_day = (hour >= self.day_start) and (hour < self.day_end)

        if is_day:
            losses[0] -= 1.0
            losses[1] += 1.0
        else:
            losses[0] += 1.0
            losses[1] -= 1.0

        return losses, self.predict(features)

class WindRegime:
    def __init__(self, wind_feature_idx, wind_mean, wind_std, strength=2.0):
        self.idx = wind_feature_idx
        self.mean = float(wind_mean)
        self.std = float(wind_std)
        self.strength = float(strength)
        self.low_th = self.mean - self.std
        self.high_th = self.mean + self.std

    def predict(self, features):
        wind = float(features[self.idx])

        if wind > self.high_th:
            return np.array([0.0, 1.0])  # high
        elif wind < self.low_th:
            return np.array([1.0, 0.0])  # low
        else:
            return np.array([0.5, 0.5])  # med

    def bias(self, losses, features):
        losses = losses.copy()
        wind = float(features[self.idx])

        if wind > self.high_th:
            losses[0] += self.strength
            losses[1] -= self.strength
        elif wind < self.low_th:
            losses[0] -= self.strength
            losses[1] += self.strength

        return losses, self.predict(features)

class Regime:
    def __init__(self, name, regimes, predictor, prior=None):
        self.name = name
        self.regimes = regimes
        self.predictor = predictor
        self.prior = prior if prior is not None else RegimePrior()

    def predict(self, features):
        return self.predictor.predict(features)

    def update(self, features, losses):
        if hasattr(self.predictor, "update"):
            losses, p_target = self.prior.bias(losses, features)
            self.predictor.update(features, losses, p_target, target_strength=5.0)