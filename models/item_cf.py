
import numpy as np
from scipy import sparse

class ItemCF:
    def __init__(self, neighbors=50, shrink=10.0):
        self.neighbors = neighbors
        self.shrink = shrink

    def fit(self, X):
        self.X = X.astype(np.float32).tocsr()
        gram = (self.X.T @ self.X).toarray()
        norms = np.sqrt(np.diag(gram).copy())
        denominator = norms[:, None] * norms[None, :] + self.shrink
        similarity = gram / np.maximum(denominator, 1e-12)
        np.fill_diagonal(similarity, 0.0)

        n_items = similarity.shape[0]
        k = min(self.neighbors, n_items - 1)
        rows = np.argpartition(-similarity, k - 1, axis=0)[:k]
        columns = np.broadcast_to(np.arange(n_items), rows.shape)
        values = similarity[rows, columns]

        self.W = sparse.csr_matrix(
            (values.ravel(), (rows.ravel(), columns.ravel())),
            shape=(n_items, n_items)
        )
        self.W.eliminate_zeros()
        return self

    def score(self, users):
        return (self.X[users] @ self.W).toarray()
