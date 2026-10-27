
import numpy as np
from scipy.linalg import solve

class EASE:
    def __init__(self, regularization=100.0):
        self.regularization = regularization

    def fit(self, X):
        self.X = X.astype(np.float32).tocsr()
        X64 = self.X.astype(np.float64)
        gram = (X64.T @ X64).toarray()
        diagonal = np.diag_indices_from(gram)
        gram[diagonal] += self.regularization

        inverse = solve(
            gram,
            np.eye(gram.shape[0]),
            assume_a="pos",
            check_finite=False
        )
        weights = -inverse / np.diag(inverse)[None, :]
        np.fill_diagonal(weights, 0.0)
        self.W = weights.astype(np.float32)
        return self

    def score(self, users):
        return np.asarray(self.X[users] @ self.W)
