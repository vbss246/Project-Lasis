import numpy as np
from sklearn import svm
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import classification_report, accuracy_score
import joblib

class SVMClassifier:
    def __init__(self, kernel='rbf', C=1.0):
        """Initialize the SVM classifier.
        
        Args:
            kernel (str): Kernel type to be used in the algorithm
            C (float): Regularization parameter
        """
        self.classifier = svm.SVC(kernel=kernel, C=C, probability=True)
        self.scaler = StandardScaler()
        self.is_trained = False
    
    def train(self, features, labels):
        """Train the SVM classifier.
        
        Args:
            features (numpy.ndarray): Feature vectors
            labels (numpy.ndarray): Target labels
        """
        # Scale features
        scaled_features = self.scaler.fit_transform(features)
        
        # Train classifier
        self.classifier.fit(scaled_features, labels)
        self.is_trained = True
    
    def predict(self, features):
        """Predict class labels for samples in features.
        
        Args:
            features (numpy.ndarray): Feature vectors
            
        Returns:
            numpy.ndarray: Predicted class labels
        """
        if not self.is_trained:
            raise ValueError("Classifier is not trained yet")
        
        # Scale features
        scaled_features = self.scaler.transform(features)
        
        # Predict
        return self.classifier.predict(scaled_features)
    
    def predict_proba(self, features):
        """Predict class probabilities for samples in features.
        
        Args:
            features (numpy.ndarray): Feature vectors
            
        Returns:
            numpy.ndarray: Class probabilities
        """
        if not self.is_trained:
            raise ValueError("Classifier is not trained yet")
        
        # Scale features
        scaled_features = self.scaler.transform(features)
        
        # Predict probabilities
        return self.classifier.predict_proba(scaled_features)
    
    def evaluate(self, features, labels):
        """Evaluate the classifier.
        
        Args:
            features (numpy.ndarray): Feature vectors
            labels (numpy.ndarray): Target labels
            
        Returns:
            dict: Evaluation results
        """
        if not self.is_trained:
            raise ValueError("Classifier is not trained yet")
        
        # Scale features
        scaled_features = self.scaler.transform(features)
        
        # Predict
        predictions = self.classifier.predict(scaled_features)
        
        # Calculate metrics
        report = classification_report(labels, predictions, output_dict=True)
        accuracy = accuracy_score(labels, predictions)
        
        return {
            'classification_report': report,
            'accuracy': accuracy
        }
    
    def save_model(self, model_path, scaler_path):
        """Save the trained model and scaler.
        
        Args:
            model_path (str): Path to save the model
            scaler_path (str): Path to save the scaler
        """
        if not self.is_trained:
            raise ValueError("Cannot save untrained model")
        
        joblib.dump(self.classifier, model_path)
        joblib.dump(self.scaler, scaler_path)
    
    @classmethod
    def load_model(cls, model_path, scaler_path):
        """Load a trained model and scaler.
        
        Args:
            model_path (str): Path to the saved model
            scaler_path (str): Path to the saved scaler
            
        Returns:
            SVMClassifier: Loaded classifier
        """
        instance = cls()
        instance.classifier = joblib.load(model_path)
        instance.scaler = joblib.load(scaler_path)
        instance.is_trained = True
        return instance