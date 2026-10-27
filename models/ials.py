
import time
import numpy as np
from scipy.linalg import solve

class ImplicitALS:
    def __init__(
        self,
        factors=32,
        regularization=1.0,
        alpha=20.0,
        iterations=8,
        seed=42
    ):
        self.factors = factors
        self.regularization = regularization
        self.alpha = alpha
        self.iterations = iterations
        self.seed = seed

    def _update(self, preferences, fixed):
        gram = fixed.T @ fixed
        base = gram + self.regularization * np.eye(self.factors)
        result = np.zeros(
            (preferences.shape[0], self.factors),
            dtype=np.float64
        )

        for row in range(preferences.shape[0]):
            start = preferences.indptr[row]
            end = preferences.indptr[row + 1]
            indices = preferences.indices[start:end]
            targets = preferences.data[start:end]

            if len(indices) == 0:
                continue

            observed = fixed[indices]
            matrix = base + self.alpha * (observed.T @ observed)
            vector = (1.0 + self.alpha) * (observed.T @ targets)

            result[row] = solve(
                matrix,
                vector,
                assume_a="pos",
                check_finite=False
            )

        return result

    def fit(self, X, targets=None):
        if self.regularization <= 0 or self.alpha < 0:
            raise ValueError("Invalid regularization or alpha.")

        observed = X.astype(np.float64).tocsr()
        observed.sort_indices()

        if targets is None:
            preferences = observed.copy()
            preferences.data[:] = 1.0
        else:
            preferences = targets.astype(np.float64).tocsr()
            preferences.sort_indices()

            if (
                preferences.shape != observed.shape
                or not np.array_equal(
                    preferences.indptr, observed.indptr
                )
                or not np.array_equal(
                    preferences.indices, observed.indices
                )
            ):
                raise ValueError(
                    "Targets must retain all observed entries, including zeros."
                )

        preferences_t = preferences.T.tocsr()
        rng = np.random.default_rng(self.seed)

        item_factors = rng.normal(
            0.0, 0.01,
            size=(observed.shape[1], self.factors)
        )

        self.history_ = []

        for epoch in range(self.iterations):
            started = time.perf_counter()
            user_factors = self._update(preferences, item_factors)
            item_factors = self._update(preferences_t, user_factors)

            elapsed = time.perf_counter() - started
            self.history_.append(elapsed)
            print(
                f"    epoch {epoch + 1}/{self.iterations}: "
                f"{elapsed:.1f} sec",
                flush=True
            )

        self.user_factors = user_factors
        self.item_factors = item_factors
        return self

    def score(self, users):
        return self.user_factors[users] @ self.item_factors.T
