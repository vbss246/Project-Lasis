import os
import torch
import numpy as np
from tqdm import tqdm
from data.dataset import get_data_loaders
from models.feature_extractor import get_feature_extractor
from models.svm_classifier import SVMClassifier

def extract_features(data_loader, feature_extractor, device):
    """Extract features from all images in the dataset.
    
    Args:
        data_loader: DataLoader instance
        feature_extractor: Feature extraction model
        device: Device to run the model on
        
    Returns:
        tuple: (features, labels)
    """
    features = []
    labels = []
    
    with torch.no_grad():
        for images, batch_labels in tqdm(data_loader, desc='Extracting features'):
            images = images.to(device)
            batch_features = feature_extractor(images)
            
            features.append(batch_features.cpu().numpy())
            labels.append(batch_labels.numpy())
    
    return np.vstack(features), np.concatenate(labels)

def train_model(data_dir, save_dir, batch_size=32, cnn_model='resnet50'):
    """Train the hybrid CNN-SVM model.
    
    Args:
        data_dir (str): Directory containing the dataset
        save_dir (str): Directory to save the trained models
        batch_size (int): Batch size for training
        cnn_model (str): Name of the CNN model to use
    """
    # Create save directory if it doesn't exist
    os.makedirs(save_dir, exist_ok=True)
    
    # Set device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f'Using device: {device}')
    
    # Get data loaders
    train_loader, val_loader = get_data_loaders(data_dir, batch_size)
    print(f'Training samples: {len(train_loader.dataset)}')
    print(f'Validation samples: {len(val_loader.dataset)}')
    
    # Initialize feature extractor
    feature_extractor = get_feature_extractor(model_name=cnn_model, device=device)
    print('Feature extractor initialized')
    
    # Extract features for training and validation sets
    print('\nExtracting features from training set...')
    train_features, train_labels = extract_features(train_loader, feature_extractor, device)
    
    print('\nExtracting features from validation set...')
    val_features, val_labels = extract_features(val_loader, feature_extractor, device)
    
    # Initialize and train SVM classifier
    print('\nTraining SVM classifier...')
    svm = SVMClassifier(kernel='rbf', C=1.0)
    svm.train(train_features, train_labels)
    
    # Evaluate the model
    print('\nEvaluating model...')
    eval_results = svm.evaluate(val_features, val_labels)
    
    # Print results
    print('\nClassification Report:')
    report = eval_results['classification_report']
    print(f'Accuracy: {report["accuracy"]:.4f}')
    print(f'Macro Avg F1-Score: {report["macro avg"]["f1-score"]:.4f}')
    
    # Save the models
    model_path = os.path.join(save_dir, 'svm_model.joblib')
    scaler_path = os.path.join(save_dir, 'feature_scaler.joblib')
    svm.save_model(model_path, scaler_path)
    print(f'\nModels saved in {save_dir}')

if __name__ == '__main__':
    # Set paths
    data_dir = '../data/crop_diseases'
    save_dir = '../models/saved'
    
    # Train the model
    train_model(data_dir, save_dir)