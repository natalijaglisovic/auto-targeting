import numpy as np
from sklearn.cluster import SpectralCoclustering


def get_bicluster_score(matrix, row_indices, col_indices, global_mean):
    """Calculate score, utility, and lift for a bicluster."""
    submatrix = matrix[np.ix_(row_indices, col_indices)]
    score = np.mean(submatrix)      # Quality (density)
    utility = np.sum(submatrix)     # Total utility
    lift = score / global_mean if global_mean > 0 else 0.0
    return score, utility, lift


def gini_coefficient(values):
    """Gini coefficient of a distribution. 0 = perfect equality, 1 = max inequality."""
    x = np.asarray(values, dtype=np.float64)
    if len(x) == 0 or np.sum(x) == 0:
        return 0.0
    x = np.sort(x)
    n = len(x)
    index = np.arange(1, n + 1)
    return (2 * np.sum(index * x) - (n + 1) * np.sum(x)) / (n * np.sum(x))


def compute_gini(matrix, biclusters):
    """
    Compute Gini coefficient of per-user utilities across all biclusters.

    Per-user utility = mean affinity between that user and the items in their
    assigned ad. Matrix is (items x users).

    :param matrix: interaction/affinity matrix, shape (n_items, n_users)
    :param biclusters: list of (row_indices, col_indices) = (item_indices, user_indices)
    :returns: Gini coefficient (float)
    """
    per_user_utils = []
    for row_indices, col_indices in biclusters:
        if len(row_indices) > 0 and len(col_indices) > 0:
            submatrix = matrix[np.ix_(row_indices, col_indices)]
            # mean over items (axis=0) gives per-user utility
            per_user_utils.append(np.mean(submatrix, axis=0))
    if len(per_user_utils) == 0:
        return 0.0
    return gini_coefficient(np.concatenate(per_user_utils))


class SpectralBicluster:
    def __init__(self, n_clusters=5):
        self.n_clusters = n_clusters
        self.model = SpectralCoclustering(n_clusters=n_clusters, random_state=0)
        self.biclusters = []

    def fit(self, matrix):
        print(f"Fitting Spectral Co-Clustering with {self.n_clusters} clusters...")
        self.model.fit(matrix)

        self.biclusters = []
        for cluster_idx in range(self.n_clusters):
            row_indices = np.where(self.model.row_labels_ == cluster_idx)[0]
            col_indices = np.where(self.model.column_labels_ == cluster_idx)[0]

            if len(row_indices) > 0 and len(col_indices) > 0:
                self.biclusters.append((row_indices, col_indices))


class ConstrainedSpectralBicluster:
    def __init__(self, random_state=0):
        self.biclusters = []
        self.random_state = random_state

    def fit(self, matrix, n_clusters=5, min_items=10, min_users=5, max_items=100, max_users=50, search_alpha=2):
        self.biclusters = []

        search_k = n_clusters * search_alpha # Change this multiplier to control how many initial clusters we generate before pruning. Higher = more candidates but slower.
        print(f"Running Spectral Co-Clustering (searching {search_k} partitions)...")
        model = SpectralCoclustering(n_clusters=search_k, random_state=self.random_state)
        model.fit(matrix)

        print("Pruning clusters to meet Min/Max constraints...")

        candidates = []

        for i in range(search_k):
            rows = np.where(model.row_labels_ == i)[0]
            cols = np.where(model.column_labels_ == i)[0]

            if len(rows) < min_items or len(cols) < min_users:
                continue

            if len(rows) > max_items:
                sub_data = matrix[np.ix_(rows, cols)]
                row_scores = np.mean(sub_data, axis=1)
                top_row_args = np.argsort(row_scores)[::-1][:max_items]
                rows = rows[top_row_args]

            if len(cols) > max_users:
                sub_data = matrix[np.ix_(rows, cols)]
                col_scores = np.mean(sub_data, axis=0)
                top_col_args = np.argsort(col_scores)[::-1][:max_users]
                cols = cols[top_col_args]

            if len(rows) >= min_items and len(cols) >= min_users:
                candidates.append((rows, cols))

        candidates.sort(key=lambda x: np.mean(matrix[np.ix_(x[0], x[1])]), reverse=True)

        self.biclusters = candidates[:n_clusters]
        print(f"  > Selected top {len(self.biclusters)} clusters satisfying constraints.")
