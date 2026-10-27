
import numpy as np
from scipy import sparse
from .ials import ImplicitALS

class RatingTargetALS(ImplicitALS):
    @staticmethod
    def target_from_rating(rating):
        return (np.asarray(rating, dtype=np.float64) - 1.0) / 4.0

    def fit(self, X, interactions):
        if not interactions["rating"].between(1, 5).all():
            raise ValueError("Ratings must be between 1 and 5.")

        values = self.target_from_rating(
            interactions["rating"].to_numpy()
        )

        # Explicit zeros are retained: a rating of 1 is still observed.
        preferences = sparse.csr_matrix(
            (
                values,
                (
                    interactions["u"].to_numpy(),
                    interactions["i"].to_numpy()
                )
            ),
            shape=X.shape
        )

        return super().fit(X, targets=preferences)
