from typing import List
import numpy as np

class knn_classifier:
    '''
    A simple k-nearest neighbors classifier.
    '''
    def __init__(self, k: int) -> None:
        '''
        Initialize the classifier.
        Parameters:
            k: The number of neighbors to consider.
        '''
        self.k = k
        self.train_features = None
        self.train_labels = None

    def fit(self, features: np.ndarray, labels: np.ndarray) -> None:
        '''
        Fit the classifier to the training data.
        Parameters:
            features: A 2D numpy array of shape (num_samples, num_features).
            labels: A 1D numpy array of shape (num_samples,).
        '''
        self.train_features = features
        self.train_labels = labels

    def predict(self, features: np.ndarray) -> List[int]:
        '''
        Predict the labels of the input features.
        Parameters:
            features: A 2D numpy array of shape (num_samples, num_features).
        Returns:
            A list of predicted labels.
        '''
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
        '''
        Compute the accuracy of the classifier.
        Parameters:
            true_labels: A 1D numpy array of shape (num_samples,).
            predicted_labels: A list of predicted labels.
        Returns:
            The accuracy of the classifier.
        '''
        return np.mean(true_labels == predicted_labels)

class knn_classifier_cross_validation(knn_classifier):
    '''
    A simple k-nearest neighbors classifier with k-fold cross-validation.
    '''
    def __init__(self, k: int, num_folds: int) -> None:
        '''
        Initialize the classifier.
        Parameters:
            k: The number of neighbors to consider.
            num_folds: The number of folds to use in cross-validation.
        '''
        super().__init__(k)
        self.num_folds = num_folds

    def cross_validate(self, features: np.ndarray, labels: np.ndarray) -> float:
        '''
        Perform k-fold cross-validation.
        Parameters:
            features: A 2D numpy array of shape (num_samples, num_features).
            labels: A 1D numpy array of shape (num_samples,).
        Returns:
            The average accuracy of the classifier.
        '''
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
    