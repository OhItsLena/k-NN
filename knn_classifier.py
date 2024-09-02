from typing import List
import numpy as np
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split

class knn_classifier:
    def __init__(self, k: int) -> None:
        self.k = k
        self.train_features = None
        self.train_labels = None

    def fit(self, features: np.ndarray, labels: np.ndarray) -> None:
        self.train_features = features
        self.train_labels = labels

    def predict(self, features: np.ndarray) -> List[int]:
        predictions = []
        for feature in features:
            distances = np.linalg.norm(self.train_features - feature, axis=1)
            nearest_indices = np.argsort(distances)[:self.k]
            nearest_labels = self.train_labels[nearest_indices]
            unique_labels, counts = np.unique(nearest_labels, return_counts=True)
            majority_label = unique_labels[np.argmax(counts)]
            predictions.append(majority_label)
        return predictions
    
    def accuracy(self, true_labels: np.ndarray, predicted_labels: List[int]) -> float:
        return np.mean(true_labels == predicted_labels)