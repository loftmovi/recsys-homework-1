import time
import numpy as np
import torch
from torch import nn


class FMNetwork(nn.Module):
    def __init__(self, n_users, n_features, factors):
        super().__init__()
        self.user_vectors = nn.Embedding(n_users, factors)
        self.user_bias = nn.Embedding(n_users, 1)
        self.feature_vectors = nn.Embedding(n_features, factors, padding_idx=0)
        self.feature_bias = nn.Embedding(n_features, 1, padding_idx=0)
        self.bias = nn.Parameter(torch.zeros(()))
        nn.init.normal_(self.user_vectors.weight, std=0.01)
        nn.init.normal_(self.feature_vectors.weight, std=0.01)
        nn.init.zeros_(self.user_bias.weight)
        nn.init.zeros_(self.feature_bias.weight)
        with torch.no_grad():
            self.feature_vectors.weight[0].zero_()

    def item_terms(self, features):
        vectors = self.feature_vectors(features)
        summed = vectors.sum(dim=1)
        interactions = 0.5 * (summed.square() - vectors.square().sum(dim=1)).sum(dim=1)
        offset = self.feature_bias(features).sum(dim=1).squeeze(-1) + interactions
        return summed, offset

    def forward(self, users, features):
        item_vectors, item_offset = self.item_terms(features)
        cross = (self.user_vectors(users) * item_vectors).sum(dim=1)
        return self.bias + self.user_bias(users).squeeze(-1) + item_offset + cross


class FMRecommender:
    def __init__(self, factors=16, learning_rate=0.01, weight_decay=1e-5,
                 epochs=5, negatives=2, batch_size=2048, seed=42,
                 use_metadata=True):
        self.factors = factors
        self.learning_rate = learning_rate
        self.weight_decay = weight_decay
        self.epochs = epochs
        self.negatives = negatives
        self.batch_size = batch_size
        self.seed = seed
        self.use_metadata = use_metadata

    def _features(self, items):
        vocabulary = {}
        rows = []
        for row in items.itertuples(index=False):
            tokens = ['item:' + str(row.item_id)]
            if self.use_metadata:
                brand = str(row.brand).strip() if row.brand else ''
                tokens.append('brand:' + (brand or '__unknown__'))
                categories = row.categories if isinstance(row.categories, (list, np.ndarray)) else []
                categories = sorted(set(str(value) for value in categories if value))
                tokens.extend('category:' + value for value in categories)
                if not categories:
                    tokens.append('category:__unknown__')
            indices = []
            for token in tokens:
                if token not in vocabulary:
                    vocabulary[token] = len(vocabulary) + 1
                indices.append(vocabulary[token])
            rows.append(indices)
        width = max(map(len, rows))
        padded = np.zeros((len(rows), width), dtype=np.int64)
        for index, values in enumerate(rows):
            padded[index, :len(values)] = values
        self.feature_vocabulary_ = vocabulary
        return torch.from_numpy(padded)

    def fit(self, X, interactions, items):
        torch.set_num_threads(2)
        torch.manual_seed(self.seed)
        rng = np.random.default_rng(self.seed)
        X = X.tocsr()
        if len(items) != X.shape[1]:
            raise ValueError('Metadata must follow the item-index order.')
        if np.any(np.diff(X.indptr) >= X.shape[1]):
            raise ValueError('A user has no unseen items for negative sampling.')
        features = self._features(items)
        self.network_ = FMNetwork(X.shape[0], len(self.feature_vocabulary_) + 1, self.factors)
        users = interactions['u'].to_numpy(dtype=np.int64)
        item_indices = interactions['i'].to_numpy(dtype=np.int64)
        labels = (interactions['rating'].to_numpy() >= 4).astype(np.float32)

        negative_users = np.repeat(users, self.negatives)
        negative_items = rng.integers(0, X.shape[1], size=len(negative_users))
        while True:
            invalid = np.asarray(X[negative_users, negative_items]).ravel() > 0
            if not invalid.any():
                break
            negative_items[invalid] = rng.integers(0, X.shape[1], size=int(invalid.sum()))

        all_users = torch.from_numpy(np.concatenate([users, negative_users]))
        all_items = torch.from_numpy(np.concatenate([item_indices, negative_items]))
        all_labels = torch.from_numpy(np.concatenate([labels, np.zeros(len(negative_users), dtype=np.float32)]))
        optimizer = torch.optim.Adam(self.network_.parameters(), lr=self.learning_rate, weight_decay=self.weight_decay)
        loss_function = nn.BCEWithLogitsLoss()
        self.history_ = []
        self.network_.train()

        for epoch in range(self.epochs):
            started = time.perf_counter()
            order = rng.permutation(len(all_users))
            total_loss = 0.0
            for start in range(0, len(order), self.batch_size):
                batch = torch.from_numpy(order[start:start + self.batch_size])
                optimizer.zero_grad(set_to_none=True)
                logits = self.network_(all_users[batch], features[all_items[batch]])
                loss = loss_function(logits, all_labels[batch])
                loss.backward()
                optimizer.step()
                total_loss += loss.item() * len(batch)
            mean_loss = total_loss / len(order)
            self.history_.append(mean_loss)
            print(f'    epoch {epoch + 1}/{self.epochs}: loss={mean_loss:.5f}, {time.perf_counter() - started:.1f} sec', flush=True)

        self.network_.eval()
        with torch.no_grad():
            item_vectors, item_offsets = self.network_.item_terms(features)
            self.item_vectors_ = item_vectors.numpy().copy()
            self.item_offsets_ = item_offsets.numpy().copy()
            self.user_vectors_ = self.network_.user_vectors.weight.numpy().copy()
            self.user_bias_ = self.network_.user_bias.weight.numpy().ravel().copy()
            self.bias_ = float(self.network_.bias)
        self.item_features_ = features
        return self

    def score(self, users):
        users = np.asarray(users, dtype=np.int64)
        return (self.user_vectors_[users] @ self.item_vectors_.T
                + self.user_bias_[users, None]
                + self.item_offsets_[None, :] + self.bias_)
