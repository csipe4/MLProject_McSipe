import numpy as np
from collections import Counter
from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import KNeighborsClassifier

# --- TASK 1: Manual Accuracy ---
def calculate_accuracy(y_true, y_pred):
    """
    Returns the ratio of correct predictions to total predictions.
    Do not use sklearn.metrics for this implementation.
    """
    # TODO: Implement manual accuracy calculation
    # turn data into arrays
    y_true, y_pred = np.array(y_true), np.array(y_pred)

    # calculate accuracy
    return np.sum(y_true == y_pred) / len(y_true)

    pass

class KNNClassifier:
    """
    A K-Nearest Neighbors classifier built from scratch.
    """
    def __init__(self, k=3, metric='euclidean'):
        self.k = k
        self.metric = metric
        self.X_train, self.y_train = None, None

    def fit(self, X, y):
        """Stores the training data."""
        # Remember in KNN there is no training, it is just data storage
        # training --> 0(1)
        # predict --> O(n)
        self.X_train, self.y_train = np.array(X), np.array(y)

    def predict(self, X_test):
        """Returns predictions for the provided test set."""
        # Process the data one by one (row by row)
        # we use a list comprehension to apply _predict_single() to every row
        return np.array([self._predict_single(x) for x in np.array(X_test)])

    def _euclidean_distance(self, row1, row2):
        """
        Calculates the L2 Norm: sqrt(sum((x_i - y_i)^2))
        """
        # TODO: Implement the Euclidean distance formula
        # standard straight-line distance
        # Pythagorean Theorem
        return np.sqrt(np.sum((row1 - row2)**2))

    def _cosine_similarity(self, row1, row2):
        """
        Calculates (u . v) / (||u|| * ||v||).
        Must handle zero division.
        """
        # TODO: Implement Dot Product / (Norm A * Norm B)
        # measure the angle between the vectors
        # NOT DISTANCE
        # best for higher-dimensional or text
        dot_product = np.dot(row1, row2)
        norm_prod = np.linalg.norm(row1) * np.linalg.norm(row2)

        return dot_product / norm_prod if norm_prod != 0 else 0.0

    def _predict_single(self, x_new):
        """
        Finds the k-nearest neighbors and returns the majority vote.
        """
        distances = []
        for i, x_train in enumerate(self.X_train):
            # TODO: Calculate distance based on self.metric. 
            # Hint: if metric is 'cosine', distance = 1 - similarity.
            if self.metric == 'euclidean':
                # lower value = closer
                dist = self._euclidean_distance(x_new, x_train)
            else: 
                # cosine return similarity (higher = closer)
                # KNN minimize the distance
                # distance = 1 - similarity
                dist = 1 - self._cosine_similarity(x_new, x_train)
            # store the distance and the actual label of that point
            distances.append((dist, self.y_train[i]))
        
        # sort by distance (small -> large) and keep the top k
        # key = lambda x: x[0] --> tell python to sort based on the distance, not the label
        k_nearest = sorted(distances, key = lambda x: x[0])[:self.k]
        
        # extract the labels (VOTE!!!)
        k_labels = [label for (_, label) in k_nearest]

        # majority vote is
        # Counter counts frequencies
        # most_common() --> the winner
        winner, count = Counter(k_labels).most_common(1)[0]
        
        return winner


# --- TASK 2: The Workflow ---


# 1. Load and Split Data
data = load_breast_cancer()
data.data[:5]
data.target[:5]
X_train, X_test, y_train, y_test = train_test_split(
    data.data, data.target, test_size=0.2, random_state=42
)

# TODO: Experiment 1 - Raw Data: Euclidean vs Cosine
print("--- Step 1: Evaluating Raw Data (Unscaled) ---")
model_euc_raw = KNNClassifier(k = 5, metric = 'euclidean')
model_euc_raw.fit(X_train, y_train) # storing data
preds_euc_raw = model_euc_raw.predict(X_test) # y_pred

model_cos_raw = KNNClassifier(k = 5, metric = 'cosine')
model_cos_raw.fit(X_train, y_train)
preds_cos_raw = model_cos_raw.predict(X_test)
# Instantiate, fit, and predict using both metrics. Print the accuracies.
print(f'Euclidean Raw Accuracy {calculate_accuracy(y_test, preds_euc_raw):.4f}%')
print(f'Cosine Raw Accuracy {calculate_accuracy(y_test, preds_cos_raw):.4f}%')


# TODO: Experiment 2 - Scaled Data: Euclidean vs Cosine
print("\n--- Step 2: Evaluating Scaled Data ---")
# Use StandardScaler to transform the data.
scaler = StandardScaler() # mean = 0, st = 1
# Re-evaluate both metrics on the scaled data and print results.
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# evaluate on scaled data
model_euc_scaled = KNNClassifier(k = 5, metric = 'euclidean')
model_euc_scaled.fit(X_train_scaled, y_train)
preds_euc_scaled = model_euc_scaled.predict(X_test_scaled)

model_cos_scaled = KNNClassifier(k = 5, metric = 'cosine')
model_cos_scaled.fit(X_train_scaled, y_train)
preds_cos_scaled = model_cos_scaled.predict(X_test_scaled)

print(f'Euclidean Scaled Accuracy: {100*calculate_accuracy(y_test, preds_euc_scaled):.4f}%')
print(f'Cosine Scaled Accuracy: {100*calculate_accuracy(y_test, preds_cos_scaled):.4f}%')


# TODO: Experiment 3 - Sklearn Comparison
print("\n--- Step 3: Sklearn Comparison ---")
# Compare your Scaled Euclidean result with sklearn's KNeighborsClassifier.
sk_knn_scaled = KNeighborsClassifier(n_neighbors = 5, metric = 'euclidean')
sk_knn_scaled.fit(X_train_scaled, y_train)
sk_preds_scaled = sk_knn_scaled.predict(X_test_scaled)

print(f'Sklearn scaled accuracy: {100*calculate_accuracy(y_test, sk_preds_scaled):.4f}%')
