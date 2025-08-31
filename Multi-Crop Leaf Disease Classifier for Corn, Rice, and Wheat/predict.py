import os
import torch
import numpy as np
from PIL import Image
from torchvision import transforms
from models.feature_extractor import get_feature_extractor
from models.svm_classifier import SVMClassifier

class DiseasePredictor:
    def __init__(self, model_dir, cnn_model='resnet50'):
        """Initialize the disease predictor.
        
        Args:
            model_dir (str): Directory containing saved models
            cnn_model (str): Name of the CNN model to use
        """
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        
        # Load feature extractor
        self.feature_extractor = get_feature_extractor(model_name=cnn_model, device=self.device)
        
        # Load SVM classifier
        svm_path = os.path.join(model_dir, 'svm_model.joblib')
        scaler_path = os.path.join(model_dir, 'feature_scaler.joblib')
        self.svm = SVMClassifier.load_model(svm_path, scaler_path)
        
        # Define image transforms
        self.transform = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406],
                             std=[0.229, 0.224, 0.225])
        ])
        
        # Load class names
        self.classes = self._load_class_names(model_dir)
    
    def _load_class_names(self, model_dir):
        """Load class names from the saved model directory.
        
        Args:
            model_dir (str): Directory containing saved models
            
        Returns:
            list: List of class names
        """
        # This should be replaced with actual class names from your dataset
        return [
            'Corn_Common_Rust',
            'Corn_Northern_Leaf_Blight',
            'Corn_Healthy',
            'Rice_Bacterial_Leaf_Blight',
            'Rice_Brown_Spot',
            'Rice_Healthy',
            'Wheat_Leaf_Rust',
            'Wheat_Septoria',
            'Wheat_Healthy'
        ]
    
    def predict_image(self, image_path):
        """Predict disease for a single image.
        
        Args:
            image_path (str): Path to the image file
            
        Returns:
            tuple: (predicted_class, confidence_score)
        """
        # Load and preprocess image
        image = Image.open(image_path).convert('RGB')
        image_tensor = self.transform(image).unsqueeze(0).to(self.device)
        
        # Extract features
        with torch.no_grad():
            features = self.feature_extractor(image_tensor)
            features = features.cpu().numpy()
        
        # Get predictions and probabilities
        pred_class = self.svm.predict(features)[0]
        probabilities = self.svm.predict_proba(features)[0]
        confidence = probabilities[pred_class]
        
        return self.classes[pred_class], confidence
    
    def predict_batch(self, image_paths):
        """Predict diseases for multiple images.
        
        Args:
            image_paths (list): List of image file paths
            
        Returns:
            list: List of (predicted_class, confidence_score) tuples
        """
        results = []
        for image_path in image_paths:
            try:
                prediction = self.predict_image(image_path)
                results.append((os.path.basename(image_path), *prediction))
            except Exception as e:
                print(f'Error processing {image_path}: {str(e)}')
                results.append((os.path.basename(image_path), 'Error', 0.0))
        return results

def main():
    # Example usage
    model_dir = '../models/saved'
    predictor = DiseasePredictor(model_dir)
    
    # Single image prediction
    image_path = '../data/test/test_image.jpg'
    predicted_class, confidence = predictor.predict_image(image_path)
    print(f'\nPrediction for {os.path.basename(image_path)}:')
    print(f'Disease: {predicted_class}')
    print(f'Confidence: {confidence:.2%}')

if __name__ == '__main__':
    main()