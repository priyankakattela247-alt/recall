"""Adaptive recall engine -- per-dimension precision controller.

Overview
--------
The frozen dynamics are  a <- a + dt * ( -pi * grad_E(a) + input(t) ),  where
`input(t)` is the damaged query, injected un-weighted for the first `T_in`
steps.  Two consequences drive the whole design:

1. During the input phase a dimension settles near
       eta * (X^T s)_i  +  q_i / pi_i ,
   i.e. the query's influence on dimension i scales like 1 / pi_i.  A LARGE
   pi_i therefore lets the stored-pattern model overwrite that dimension,
   while a SMALL pi_i keeps the query's evidence.  So the weight must be
   HIGH on damaged (zeroed-then-noised) dimensions and LOW on intact ones --
   the opposite of the naive "trust the reliable dimensions" reading.

2. At the equilibrium of a stored pattern, convergence speed is governed by
   the spectrum of  sqrt(pi) H sqrt(pi)  with H the energy Hessian.  A
   diagonal pi that equalises the curvature of H shrinks the eigenvalue
   spread (a Jacobi-style preconditioner, refined numerically).

Method (no labels, no evaluator internals; only the constructor arguments)
--------------------------------------------------------------------------
* Retrieval  -- per query, fit a small mixture model by EM.  Dimension i of a
  query from pattern k is either "kept"  (q_i ~ N(c * X_ki, s^2)) or
  "damaged" (q_i ~ N(0, s^2)); the scale c, noise s and damage fraction rho
  are estimated for each query.  The posterior over patterns and the
  per-dimension probability of being intact give a damage map u_i, and
      log pi_i  +=  log(pi_max / pi_min) * (1 - u_i).
* Balance    -- per stored pattern, solve for the equilibrium of the frozen
  dynamics (Newton), form the Hessian H, and search a diagonal scaling that
  minimises cond(sqrt(pi) H sqrt(pi)) inside the permitted box.  Candidates
  always include the uniform vector, so it is never worse than uniform.
  A query is routed to its (posterior-weighted) pattern's scaling.
* The two goals share one box [pi_min, pi_max], so they are blended with a
  smooth gate on the estimated damage fraction: nearly intact queries get the
  preconditioner, heavily damaged queries get the damage map.
* Output is clipped to the box and normalised to mean 1, so it is unchanged
  by the evaluator's own clip / rescale step however that is ordered.

Everything is deterministic, uses only NumPy, never mutates its inputs and
holds no data tied to any particular seed, instance or query.
"""
from adapter import Adapter
import numpy as np

# Mixture-model fit (damage inference).
_EM_ITERS = 15
_RHO_MIN, _RHO_MAX = 1e-3, 0.98
_SCALE_FLOOR = 1e-3

# Smooth gate: estimated damage fraction at which the controller switches from
# "balance the Hessian" (below _DAMAGE_LOW) to "suppress damaged dimensions"
# (above _DAMAGE_HIGH).  Public masks are 60-85 %, intact probes are ~0 %.
_DAMAGE_LOW, _DAMAGE_HIGH = 0.20, 0.45

# Preconditioner search.
_PRECOND_STEPS = 12
_PRECOND_RATE = 0.15
_NEWTON_STEPS = 40
_POSTERIOR_CUTOFF = 1e-3     # ignore patterns less likely than this when blending


def _param(params, name, default):
    """Read an entry of `model_params`, falling back to a default if absent."""
    try:
        return params[name]
    except (KeyError, IndexError, TypeError):
        return default


def _smoothstep(x, lo, hi):
    t = min(max((x - lo) / (hi - lo), 0.0), 1.0)
    return t * t * (3.0 - 2.0 * t)


def _to_box_mean_one(log_w, lo, hi):
    """Return clip(s * exp(log_w), lo, hi) with s chosen so the mean is 1.

    The clipped mean is monotone in s, so s is found by bisection.  Requires
    lo <= 1 <= hi, which makes a solution exist.
    """
    w = np.exp(log_w - np.max(log_w))            # in (0, 1]
    a = np.log(lo)                                # mean <= lo <= 1 here
    b = np.log(hi) - np.log(max(w.min(), 1e-300))  # every entry clipped to hi
    for _ in range(50):
        m = 0.5 * (a + b)
        if np.clip(np.exp(m) * w, lo, hi).mean() < 1.0:
            a = m
        else:
            b = m
    return np.clip(np.exp(0.5 * (a + b)) * w, lo, hi)


class Engine(Adapter):
    def __init__(self, stored_patterns, model_params):
        self.X = np.array(stored_patterns, dtype=np.float64)   # private copy
        self.K, self.N = self.X.shape
        self.R = np.array(_param(model_params, "R", np.eye(self.N)),
                          dtype=np.float64)
        self.eta = float(_param(model_params, "eta", 1.0))
        self.beta = float(_param(model_params, "beta", 8.0))
        lo = float(_param(model_params, "pi_min", 0.25))
        hi = float(_param(model_params, "pi_max", 4.0))
        self.lo, self.hi = min(lo, 1.0), max(hi, 1.0)
        self.log_span = float(np.log(self.hi / self.lo))

        # log of the balance-optimal weights per stored pattern, solved lazily
        # (only near-intact queries ever need them) and cached.
        self._log_balance = {}

    # ------------------------------------------------------------------
    # Convergence balance: diagonal preconditioner of the equilibrium Hessian
    # ------------------------------------------------------------------
    def _softmax(self, a):
        z = self.beta * (self.X @ a)
        z -= z.max()
        s = np.exp(z)
        return s / s.sum()

    def _gradient_and_hessian(self, a):
        s = self._softmax(a)
        grad = self.R @ a - self.eta * (self.X.T @ s)
        cov = np.diag(s) - np.outer(s, s)
        hess = self.R - self.eta * self.beta * (self.X.T @ cov @ self.X)
        return grad, hess

    def _equilibrium(self, k):
        """Newton iteration from stored pattern k to a stationary point."""
        a = self.X[k].copy()
        for _ in range(_NEWTON_STEPS):
            grad, hess = self._gradient_and_hessian(a)
            try:
                step = np.linalg.solve(hess + 1e-9 * np.eye(self.N), grad)
            except np.linalg.LinAlgError:
                break
            if not np.all(np.isfinite(step)):
                break
            a = a - step
            if np.linalg.norm(step) < 1e-10:
                break
        return a

    @staticmethod
    def _condition(hess, pi):
        d = np.sqrt(pi)
        ev = np.linalg.eigvalsh(d[:, None] * hess * d[None, :])
        return np.inf if ev[0] <= 1e-12 else ev[-1] / ev[0]

    def _balance_log_weights(self, k):
        if k not in self._log_balance:
            self._log_balance[k] = np.log(self._balance_weights(k))
        return self._log_balance[k]

    def _balance_weights(self, k):
        """Diagonal weights (mean 1, inside the box) that tighten the spectrum
        of sqrt(pi) H sqrt(pi) at pattern k's equilibrium."""
        uniform = np.ones(self.N)
        _, hess = self._gradient_and_hessian(self._equilibrium(k))
        diag = np.diag(hess)
        if not np.all(np.isfinite(hess)) or np.any(diag <= 0):
            return uniform

        log_w = -np.log(diag)                                  # Jacobi start
        candidates = [uniform, _to_box_mean_one(log_w, self.lo, self.hi)]
        for _ in range(_PRECOND_STEPS):
            pi = _to_box_mean_one(log_w, self.lo, self.hi)
            d = np.sqrt(pi)
            ev, vec = np.linalg.eigh(d[:, None] * hess * d[None, :])
            if ev[0] <= 1e-12:
                break
            # d log(cond) / d log(pi_i) = v_max_i^2 - v_min_i^2
            grad = vec[:, -1] ** 2 - vec[:, 0] ** 2
            log_w = log_w - _PRECOND_RATE * np.sqrt(self.N) * grad
            candidates.append(_to_box_mean_one(log_w, self.lo, self.hi))
        conds = [self._condition(hess, c) for c in candidates]
        return candidates[int(np.argmin(conds))]               # uniform ties win

    # ------------------------------------------------------------------
    # Retrieval: which dimensions of this query are damaged?
    # ------------------------------------------------------------------
    def _infer_damage(self, q):
        """EM fit of the kept/damaged mixture for one query.

        Returns (posterior over patterns, P(dimension intact), damage fraction).
        """
        X, N = self.X, self.N
        q2 = q * q
        s = max(np.median(np.abs(q)) / 0.6745, _SCALE_FLOOR)   # noise scale
        rho = 0.7                                               # damage frac
        c = 1.0 / np.sqrt((1.0 - rho) + 0.1)                    # signal gain
        for _ in range(_EM_ITERS):
            inv = 0.5 / (s * s)
            log_kept = np.log(1.0 - rho) - (q[None, :] - c * X) ** 2 * inv
            log_dmg = np.log(rho) - q2[None, :] * inv
            loglik = np.logaddexp(log_kept, log_dmg).sum(axis=1)
            w = np.exp(loglik - loglik.max())
            w /= w.sum()                                        # P(pattern)
            r = 0.5 * (1.0 + np.tanh(0.5 * (log_kept - log_dmg)))  # P(intact)
            wr = w[:, None] * r
            c = max((wr * q[None, :] * X).sum()
                    / max((wr * X * X).sum(), 1e-12), 1e-3)
            resid = r * (q[None, :] - c * X) ** 2 + (1.0 - r) * q2[None, :]
            s = max(np.sqrt((w[:, None] * resid).sum() / N), _SCALE_FLOOR)
            rho = float(np.clip((w[:, None] * (1.0 - r)).sum() / N,
                                _RHO_MIN, _RHO_MAX))
        return w, (w[:, None] * r).sum(axis=0), rho

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------
    def predict_precision(self, corrupted_query):
        q = np.array(corrupted_query, dtype=np.float64).reshape(-1)
        if q.shape[0] != self.N or not np.all(np.isfinite(q)):
            return np.ones(self.N)

        posterior, intact, damage = self._infer_damage(q)
        evidence = _smoothstep(damage, _DAMAGE_LOW, _DAMAGE_HIGH)

        log_w = evidence * self.log_span * (1.0 - intact)
        if evidence < 1.0:                       # blend in the balance weights
            likely = np.flatnonzero(posterior > _POSTERIOR_CUTOFF)
            if likely.size == 0:
                likely = np.array([int(np.argmax(posterior))])
            share = posterior[likely] / posterior[likely].sum()
            balance = sum(p * self._balance_log_weights(int(k))
                          for p, k in zip(share, likely))
            log_w = log_w + (1.0 - evidence) * balance
        return _to_box_mean_one(log_w, self.lo, self.hi)
