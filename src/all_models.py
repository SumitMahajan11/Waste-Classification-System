"""
All CNN Models for Waste Classification
This file combines all individual model definition files into one.
"""

import torch
import torch.nn as nn
from torchvision import models

# EfficientNet B0 Model
def get_efficientnet_b0(num_classes, pretrained=True):
    """
    Create an EfficientNet B0 model for waste classification
    
    Args:
        num_classes (int): Number of classes for classification
        pretrained (bool): Whether to use pretrained weights
        
    Returns:
        model: EfficientNet B0 model with modified classifier
    """
    # Load pretrained EfficientNet B0
    model = models.efficientnet_b0(pretrained=pretrained)
    
    # Replace the classifier with a new one for our number of classes
    # EfficientNet B0 has 1280 features in its classifier
    in_features = 1280  # Known value for EfficientNet B0
    model.classifier = nn.Sequential(
        nn.Dropout(0.2),
        nn.Linear(in_features, num_classes)
    )
    
    return model

# DenseNet121 Model
def get_densenet121(num_classes, pretrained=True):
    """
    Create a DenseNet121 model for waste classification
    
    Args:
        num_classes (int): Number of classes for classification
        pretrained (bool): Whether to use pretrained weights
        
    Returns:
        model: DenseNet121 model with modified classifier
    """
    # Load pretrained DenseNet121
    model = models.densenet121(pretrained=pretrained)
    
    # Replace the classifier with a new one for our number of classes
    # DenseNet121 has 1024 features in its classifier
    in_features = 1024  # Known value for DenseNet121
    model.classifier = nn.Linear(in_features, num_classes)
    
    return model

# MobileNetV2 Model
def get_mobilenet_v2(num_classes, pretrained=True):
    """
    Create a MobileNetV2 model for waste classification
    
    Args:
        num_classes (int): Number of classes for classification
        pretrained (bool): Whether to use pretrained weights
        
    Returns:
        model: MobileNetV2 model with modified classifier
    """
    # Load pretrained MobileNetV2
    model = models.mobilenet_v2(pretrained=pretrained)
    
    # Replace the classifier with a new one for our number of classes
    # MobileNetV2 has 1280 features in its classifier
    in_features = 1280  # Known value for MobileNetV2
    model.classifier = nn.Sequential(
        nn.Dropout(0.2),
        nn.Linear(in_features, num_classes)
    )
    
    return model

# ResNet50 Model
def get_resnet50(num_classes, pretrained=True):
    """
    Create a ResNet50 model for waste classification
    
    Args:
        num_classes (int): Number of classes for classification
        pretrained (bool): Whether to use pretrained weights
        
    Returns:
        model: ResNet50 model with modified classifier
    """
    # Load pretrained ResNet50
    model = models.resnet50(pretrained=pretrained)
    
    # Replace the classifier with a new one for our number of classes
    # ResNet50 has 2048 features in its fc layer
    in_features = 2048  # Known value for ResNet50
    model.fc = nn.Linear(in_features, num_classes)
    
    return model

