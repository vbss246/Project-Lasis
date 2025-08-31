import torch
from torchvision import models

class FeatureExtractor(torch.nn.Module):
    def __init__(self, model_name='resnet50', pretrained=True):
        """Initialize the feature extractor.
        
        Args:
            model_name (str): Name of the pre-trained model to use
            pretrained (bool): Whether to use pre-trained weights
        """
        super(FeatureExtractor, self).__init__()
        
        if model_name == 'resnet50':
            model = models.resnet50(pretrained=pretrained)
            self.feature_dim = 2048
        elif model_name == 'vgg16':
            model = models.vgg16(pretrained=pretrained)
            self.feature_dim = 4096
        else:
            raise ValueError(f"Unsupported model: {model_name}")
        
        # Remove the classification layer
        if model_name == 'resnet50':
            self.features = torch.nn.Sequential(*(list(model.children())[:-1]))
        elif model_name == 'vgg16':
            self.features = torch.nn.Sequential(*(list(model.children())[:-1]))
        
        # Freeze the parameters
        for param in self.features.parameters():
            param.requires_grad = False
    
    def forward(self, x):
        """Extract features from the input.
        
        Args:
            x (torch.Tensor): Input tensor
            
        Returns:
            torch.Tensor: Extracted features
        """
        x = self.features(x)
        x = torch.flatten(x, 1)
        return x

def get_feature_extractor(model_name='resnet50', device='cpu'):
    """Get a feature extractor model.
    
    Args:
        model_name (str): Name of the pre-trained model to use
        device (str): Device to run the model on
        
    Returns:
        FeatureExtractor: Feature extractor model
    """
    model = FeatureExtractor(model_name=model_name)
    model = model.to(device)
    model.eval()
    return model