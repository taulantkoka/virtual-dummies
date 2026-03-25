"""
AD_LARS — Augmented-Dummy LARS (explicit dummy columns).

Pure-Python LARS with incremental Cholesky updates. Used as the AD baseline
in experiments comparing against VD-LARS (C++ vd_selectors).

Features:
  - Warm-start: call run(T=1), then run(T=5) to continue
  - Stop by dummy count or by LARS step count
  - Optional orthonormal basis tracking (for universality diagnostics, Fig 5)

Requires: helpers.pyx (Cython) or helpers.py (pure-Python fallback)
"""

import numpy as np
import logging
import sys
import os
from scipy.linalg import solve_triangular

from helpers import cholesky_rank1_update, compute_stepsize_gamma


class AD_LARS:
    """
    Vanilla LARS (no Lasso drops) on an augmented design [X | D].

    The last `num_dummies` columns of X are treated as dummies.
    Stopping is controlled by the `stop` parameter in `run()`.

    Parameters
    ----------
    X : (n, p_real + num_dummies) array, F-order recommended
    y : (n,) response vector
    num_dummies : int
        Number of trailing columns that are dummies.
    track_basis : bool
        If True, maintain an orthonormal basis in R^n that grows with
        each newly selected column (needed for Fig 5 diagnostics).
    """

    def __init__(self, X, y, num_dummies, *,
                 max_steps=None,
                 normalize=False,
                 eps=1e-12,
                 verbose=False,
                 track_basis=False):

        # --- logging ---
        self.logger = logging.getLogger(f"{__name__}.AD_LARS")
        level = logging.DEBUG if verbose else logging.WARNING
        self.logger.setLevel(level)

        for h in list(self.logger.handlers):
            if isinstance(h, logging.StreamHandler):
                self.logger.removeHandler(h)

        handler = logging.StreamHandler(stream=sys.stdout)
        handler.setLevel(level)
        handler.setFormatter(logging.Formatter("[%(levelname)s] %(message)s"))
        self.logger.addHandler(handler)
        self.logger.propagate = False

        # --- data ---
        self.X = np.asarray(X, dtype=float, order="F")
        self.y = np.asarray(y, dtype=float).copy()
        self.n, self.p_total = self.X.shape

        self.num_dummies = int(num_dummies)
        if not (0 <= self.num_dummies <= self.p_total):
            raise ValueError("num_dummies must be between 0 and p_total.")

        self.p_real = self.p_total - self.num_dummies
        self.max_steps = int(max_steps) if max_steps is not None else min(self.n, self.p_total)
        self.normalize = bool(normalize)
        self.eps = float(eps)

        # per-column entry step (-1 = never entered)
        self.entry_step = -np.ones(self.p_total, dtype=int)

        # scaling/centering
        self.normx = np.ones(self.p_total, dtype=float)
        if self.normalize:
            self.logger.debug("Normalizing X columns")
            self.normx = np.linalg.norm(self.X, axis=0)
            safe = self.normx > self.eps * np.sqrt(self.n)
            self.normx[~safe] = self.eps * np.sqrt(self.n)
            self.X = self.X / self.normx

        # --- core state ---
        self.beta = np.zeros(self.p_total)
        self.mu = np.zeros(self.n)
        self.residuals = self.y.copy()
        self.corr = self.X.T @ self.residuals

        # active set
        self.actives = []
        self._actives_set = set()
        self.signs = np.empty(0, dtype=float)

        # incremental signed active block and Cholesky factor
        self.Xp_A = np.empty((self.n, 0), dtype=float, order="F")
        self.R = None

        # dummy stopping bookkeeping
        self.selected_dummies = []
        self.selected_num_dummies = 0
        self.beta_dict = {}
        self.step = 0

        # numerics
        self.tiny32 = np.finfo(np.float32).tiny
        self._gammas = np.empty(self.p_total, dtype=float)

        # --- optional basis tracking ---
        self.track_basis = bool(track_basis)
        if self.track_basis:
            self.basis = np.empty((self.n, 0), dtype=float, order="F")
            self._init_basis_with_y()
        else:
            self.basis = None

        # initialize path with beta=0
        self.beta_path = [(self.beta / self.normx).copy()]

    # =====================================================================
    # Basis tracking (only active when track_basis=True)
    # =====================================================================

    def _init_basis_with_y(self):
        yn = float(np.linalg.norm(self.y))
        if yn <= 100 * self.eps:
            self.basis = np.empty((self.n, 0), dtype=float, order="F")
            return
        e0 = (self.y / yn).reshape(self.n, 1)
        self.basis = np.asfortranarray(e0)

    def _orthonormalize_and_add(self, v):
        v = np.asarray(v, dtype=float).reshape(self.n)
        if self.basis.shape[1] > 0:
            proj = self.basis.T @ v
            v = v - self.basis @ proj
        nv = float(np.linalg.norm(v))
        if nv <= 100 * self.eps:
            return False
        v = (v / nv).reshape(self.n, 1)
        # re-orthogonalization pass
        if self.basis.shape[1] > 0:
            proj2 = self.basis.T @ v[:, 0]
            v[:, 0] -= self.basis @ proj2
            nv2 = float(np.linalg.norm(v[:, 0]))
            if nv2 <= 100 * self.eps:
                return False
            v[:, 0] /= nv2
        self.basis = np.asfortranarray(np.column_stack([self.basis, v]))
        return True

    def get_basis(self):
        """Return a copy of the current orthonormal basis (n, q), or None."""
        return self.basis.copy() if self.basis is not None else None

    # =====================================================================
    # Helpers
    # =====================================================================

    def _is_dummy_col(self, j):
        return j >= self.p_real

    def _count_new_dummy_entries(self, new_idxs):
        for j in new_idxs:
            if self._is_dummy_col(j):
                self.selected_num_dummies += 1
                self.selected_dummies.append(int(j))
                self.beta_dict[self.selected_num_dummies] = {
                    "beta": (self.beta / self.normx).copy()
                }
                self.logger.debug(
                    "Dummy entered (index %d). Total selected dummies: %d",
                    j, self.selected_num_dummies,
                )

    def _add_actives_incremental(self, new_idxs):
        if not new_idxs:
            return
        for j in new_idxs:
            j = int(j)
            if j in self._actives_set:
                continue
            cj = float(self.corr[j])
            s = 1.0 if cj >= 0 else -1.0
            x_new_signed = self.X[:, j] * s

            if self.R is None:
                r00 = np.sqrt(float(x_new_signed @ x_new_signed))
                if r00 < 100 * self.eps:
                    self.logger.debug("Skipping near-zero column at init: %d", j)
                    continue
                self.R = np.array([[r00]], dtype=float)
                self.Xp_A = x_new_signed.reshape(self.n, 1)
            else:
                self.R = cholesky_rank1_update(self.R, self.Xp_A, x_new_signed)
                self.Xp_A = np.column_stack((self.Xp_A, x_new_signed))

            self.actives.append(j)
            self._actives_set.add(j)
            self.signs = np.append(self.signs, s)
            if self.entry_step[j] < 0:
                self.entry_step[j] = self.step

            if self.track_basis:
                self._orthonormalize_and_add(self.X[:, j])

    def _current_C_and_ties(self):
        C = float(np.max(np.abs(self.corr))) if self.corr.size else 0.0
        if C < 100 * self.eps:
            raise StopIteration
        cand = np.where(np.abs(self.corr) >= C - self.eps)[0]
        return C, cand

    # =====================================================================
    # Core LARS
    # =====================================================================

    def _compute_direction(self):
        if self.R is None or self.Xp_A.shape[1] == 0:
            raise RuntimeError("No actives: cannot compute direction.")
        ones = np.ones(self.R.shape[0])
        z = solve_triangular(self.R.T, ones, lower=True)
        w0 = solve_triangular(self.R, z, lower=False)
        A_active = 1.0 / np.sqrt(float(ones @ w0))
        w = A_active * w0
        u = self.Xp_A @ w
        a = self.X.T @ u
        return A_active, w, u, a

    def _take_step(self, C, A_active, w, u, a):
        d = np.zeros(self.p_total)
        if self.actives:
            actives_np = np.array(self.actives, dtype=np.int32)
            d[actives_np] = self.signs * w
        else:
            actives_np = np.empty(0, dtype=np.int32)

        self._gammas.fill(np.inf)
        gammas = compute_stepsize_gamma(
            self.corr, a, A_active, C,
            self.tiny32, self._gammas, actives_np,
        )
        gamma = float(np.min(gammas))
        if not np.isfinite(gamma) or gamma < self.eps:
            raise StopIteration

        self.mu += gamma * u
        self.beta += gamma * d
        self.residuals = self.y - self.mu
        self.corr -= gamma * a
        self.beta_path.append((self.beta / self.normx).copy())
        return gamma

    # =====================================================================
    # Main loop (warm-start capable)
    # =====================================================================

    def run(self, T=1, *, stop="dummies"):
        """
        Run the LARS path until a stopping criterion is met.

        Parameters
        ----------
        T : int
            Target count for the stopping rule.
        stop : {"dummies", "steps"}
            "dummies" — stop once T dummies have entered (default).
            "steps"   — stop after T completed LARS moves.
        """
        T = int(T)
        if T < 0:
            raise ValueError("T must be >= 0.")
        stop = str(stop).lower()
        if stop not in {"dummies", "steps"}:
            raise ValueError("stop must be 'dummies' or 'steps'.")

        # quick return if already satisfied
        if stop == "dummies" and self.selected_num_dummies >= T:
            return np.array(self.beta_path)
        if stop == "steps" and self.step >= T:
            return np.array(self.beta_path)

        while self.step < self.max_steps:
            if stop == "steps" and self.step >= T:
                break

            try:
                C, cand = self._current_C_and_ties()
            except StopIteration:
                break

            new_actives = [int(j) for j in cand if j not in self._actives_set]
            if new_actives:
                self.logger.debug("Knot %d: adding actives %s", self.step, new_actives)

            self._add_actives_incremental(new_actives)
            self._count_new_dummy_entries(new_actives)

            if stop == "dummies" and self.selected_num_dummies >= T:
                self.logger.debug("Stopping: reached T=%d dummies.", T)
                break

            if len(self.actives) == 0 or len(self.actives) >= self.p_total:
                break

            try:
                A_active, w, u, a = self._compute_direction()
                self._take_step(C, A_active, w, u, a)
            except StopIteration:
                break

            self.step += 1

            if stop == "steps" and self.step >= T:
                break

        return np.array(self.beta_path)