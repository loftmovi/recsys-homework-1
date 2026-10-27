
import numpy as np

def make_truth(frame, threshold=4):
    positive = frame[frame["rating"] >= threshold]
    return {
        int(user): set(group["i"].astype(int))
        for user, group in positive.groupby("u", sort=True)
    }

def recommend(model, users, seen, k=10):
    scores = np.asarray(model.score(users), dtype=np.float64).copy()
    if scores.shape != (len(users), seen.shape[1]):
        raise ValueError("Unexpected score matrix shape.")
    if not np.isfinite(scores).all():
        raise ValueError("Model produced NaN or infinity.")

    batch_seen = seen[users].tocsr()
    row_indices, item_indices = batch_seen.nonzero()
    scores[row_indices, item_indices] = -np.inf

    # Stable sorting: equal scores are ordered by item index.
    return np.argsort(-scores, axis=1, kind="stable")[:, :k]

def evaluate(model, truth, seen, k=10, batch_size=128):
    users = np.array(sorted(truth), dtype=np.int64)
    if len(users) == 0:
        raise ValueError("No users with relevant items.")

    discounts = 1.0 / np.log2(np.arange(2, k + 2))
    recalls, ndcgs = [], []

    for start in range(0, len(users), batch_size):
        batch = users[start:start + batch_size]
        recommendations = recommend(model, batch, seen, k=k)

        for user, ranked in zip(batch, recommendations):
            relevant = truth[int(user)]
            hits = np.array(
                [int(item) in relevant for item in ranked],
                dtype=np.float64
            )
            recalls.append(hits.sum() / len(relevant))
            ideal = discounts[:min(k, len(relevant))].sum()
            ndcgs.append(float(hits @ discounts) / ideal)

    return {
        "NDCG@10": float(np.mean(ndcgs)),
        "Recall@10": float(np.mean(recalls)),
        "evaluated_users": len(users)
    }
