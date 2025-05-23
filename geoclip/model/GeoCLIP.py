import os
import torch
import torch.nn as nn
import numpy as np
import torch.nn.functional as F
from .image_encoder import ImageEncoder
from .location_encoder import LocationEncoder
from .misc import load_gps_data, file_dir

from PIL import Image
from torchvision.transforms import ToPILImage

class GeoCLIP(nn.Module):
    def __init__(self, from_pretrained=True, queue_size=4096):
        super().__init__()
        self.logit_scale = nn.Parameter(torch.ones([]) * np.log(1 / 0.07))
        self.image_encoder = ImageEncoder()
        self.location_encoder = LocationEncoder()

        self.gps_gallery = load_gps_data(os.path.join(file_dir, "gps_gallery", "coordinates_100K.csv"))
        self._initialize_gps_queue(queue_size)

        if from_pretrained:
            self.weights_folder = os.path.join(file_dir, "weights")
            self._load_weights()

        self.device = "cpu"

    def to(self, device):
        self.device = device
        self.image_encoder.to(device)
        self.location_encoder.to(device)
        self.logit_scale.data = self.logit_scale.data.to(device)
        return super().to(device)

    def _load_weights(self):
        self.image_encoder.mlp.load_state_dict(torch.load(f"{self.weights_folder}/image_encoder_mlp_weights.pth", weights_only=True))
        self.location_encoder.load_state_dict(torch.load(f"{self.weights_folder}/location_encoder_weights.pth", weights_only=True))
        self.logit_scale = nn.Parameter(torch.load(f"{self.weights_folder}/logit_scale_weights.pth", weights_only=True))

    def _initialize_gps_queue(self, queue_size):
        self.queue_size = queue_size
        self.register_buffer("gps_queue", torch.randn(2, self.queue_size))
        self.gps_queue = nn.functional.normalize(self.gps_queue, dim=0)
        self.register_buffer("gps_queue_ptr", torch.zeros(1, dtype=torch.long))

    @torch.no_grad()
    def dequeue_and_enqueue(self, gps):
        """ Update GPS queue

        Args:
            gps (torch.Tensor): GPS tensor of shape (batch_size, 2)
        """
        gps_batch_size = gps.shape[0]
        gps_ptr = int(self.gps_queue_ptr)
        
        assert self.queue_size % gps_batch_size == 0, f"Queue size {self.queue_size} should be divisible by batch size {gps_batch_size}"

        # Replace the GPS from ptr to ptr+gps_batch_size (dequeue and enqueue)
        self.gps_queue[:, gps_ptr:gps_ptr + gps_batch_size] = gps.t()
        gps_ptr = (gps_ptr + gps_batch_size) % self.queue_size  # move pointer
        self.gps_queue_ptr[0] = gps_ptr

    def get_gps_queue(self):
        return self.gps_queue.t()
                                             
    def forward(self, image, location):
        """ GeoCLIP's forward pass

        Args:
            image (torch.Tensor): Image tensor of shape (n, 3, 224, 224)
            location (torch.Tensor): GPS location tensor of shape (m, 2)

        Returns:
            logits_per_image (torch.Tensor): Logits per image of shape (n, m)
        """
        # Compute Features
        image_features = self.image_encoder(image)
        location_features = self.location_encoder(location)
        logit_scale = self.logit_scale.exp()
        
        # Normalize features while preserving gradients
        image_features = F.normalize(image_features, dim=1)
        location_features = F.normalize(location_features, dim=1)
        
        # Compute similarity while preserving gradients
        logits_per_image = logit_scale * torch.matmul(image_features, location_features.t())

        return logits_per_image

    @torch.no_grad()
    def predict(self, image_path, top_k):
        """ Given an image, predict the top k GPS coordinates

        Args:
            image_path (str): Path to the image
            top_k (int): Number of top predictions to return

        Returns:
            top_pred_gps (torch.Tensor): Top k GPS coordinates of shape (k, 2)
            top_pred_prob (torch.Tensor): Top k GPS probabilities of shape (k,)
        """
        image = Image.open(image_path)
        image = self.image_encoder.preprocess_image(image)
        image = image.to(self.device)

        gps_gallery = self.gps_gallery.to(self.device)

        logits_per_image = self.forward(image, gps_gallery)
        probs_per_image = logits_per_image.softmax(dim=-1).cpu()

        # Get top k predictions
        top_pred = torch.topk(probs_per_image, top_k, dim=1)
        top_pred_gps = self.gps_gallery[top_pred.indices[0]]
        top_pred_prob = top_pred.values[0]

        return top_pred_gps, top_pred_prob
    
    def predict_logits(self, image_tensor):
        # Ensure proper dimensions (add batch dimension if needed)
        if len(image_tensor.shape) == 3:
            image_tensor = image_tensor.unsqueeze(0)
            
        # Move to the correct device
        image_tensor = image_tensor.to(self.device)
        
        gps_gallery = self.gps_gallery.to(self.device)
        
        logits_per_image = self.forward(image_tensor, gps_gallery)
        return logits_per_image
    
    
    def predict_from_tensor(self, image_tensor, top_k=1, apply_transforms=False):
        """Given an image tensor, predict the top k GPS coordinates
        
        Args:
            image_tensor (torch.Tensor): Image tensor of shape (3, H, W) or (B, 3, H, W)
            top_k (int): Number of top predictions to return
            apply_transforms (bool): Whether to apply required transformations (resize, crop, normalize)
                                   Set to True for raw tensors, False for already preprocessed tensors
            
        Returns:
            top_pred_gps (torch.Tensor): Top k GPS coordinates of shape (k, 2)
            top_pred_prob (torch.Tensor): Top k GPS probabilities of shape (k,)
        """
        # Ensure proper dimensions (add batch dimension if needed)
        if len(image_tensor.shape) == 3:
            image_tensor = image_tensor.unsqueeze(0)
            
        # Move to the correct device while preserving gradients
        if not image_tensor.is_cuda:
            image_tensor = image_tensor.to(self.device)
        
        # Apply transformations if needed
        if apply_transforms:
            from torchvision import transforms
            
            # Create transformation pipeline matching what CLIP expects
            transform = transforms.Compose([
                transforms.Resize(256),
                transforms.CenterCrop(224),
                transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225))
            ])
            
            # Apply transforms while maintaining gradients
            b, c, h, w = image_tensor.shape
            transformed_tensors = []
            
            for i in range(b):
                img = image_tensor[i]
                # Apply transforms that work directly on tensors
                transformed = transform(img)
                transformed_tensors.append(transformed)
                
            image_tensor = torch.stack(transformed_tensors)
        
        # Move GPS gallery to device and ensure it's properly formatted
        gps_gallery = self.gps_gallery.to(self.device)
        
        # Forward pass with gradient tracking
        logits_per_image = self.forward(image_tensor, gps_gallery)
        
        # Convert logits to probabilities while maintaining gradients
        probs_per_image = F.softmax(logits_per_image, dim=-1)
        
        # Get top k predictions
        top_pred_values, top_pred_indices = torch.topk(probs_per_image, top_k, dim=1)
        
        # For clean predictions, use direct indexing
        top_pred_gps = gps_gallery[top_pred_indices]
        
        # For attack mode (when requires_grad is True), compute weighted coordinates
        if image_tensor.requires_grad:
            # Compute weighted sum of all GPS coordinates
            weighted_gps = torch.matmul(probs_per_image, gps_gallery)
            top_pred_gps = weighted_gps  # Use weighted coordinates for gradient flow
        
        if top_k == 1:
            top_pred_gps = top_pred_gps.squeeze(1)  # removes the singleton top_k dimension
            top_pred_values = top_pred_values.squeeze(1)  # similarly for the probabilities

        return top_pred_gps, top_pred_values