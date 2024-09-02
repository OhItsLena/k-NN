from typing import List
import numpy as np

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

class knn_classifier_cross_validation(knn_classifier):
    def __init__(self, k: int, num_folds: int) -> None:
        super().__init__(k)
        self.num_folds = num_folds

    def cross_validate(self, features: np.ndarray, labels: np.ndarray) -> float:
        fold_size = len(labels) // self.num_folds
        accuracies = []
        for fold in range(self.num_folds):
            start = fold * fold_size
            end = start + fold_size
            validation_features = features[start:end]
            validation_labels = labels[start:end]
            training_features = np.concatenate([features[:start], features[end:]])
            training_labels = np.concatenate([labels[:start], labels[end:]])
            self.fit(training_features, training_labels)
            predicted_labels = self.predict(validation_features)
            accuracies.append(self.accuracy(validation_labels, predicted_labels))
        return np.mean(accuracies)
    