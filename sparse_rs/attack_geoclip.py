import torch
from rs_attacks import RSAttack
from sparse_rs.util import haversine_distance

class GeoCLIPPredictor():
    """
    Wraps a GeoCLIP model so that given a batch of image tensors it returns predicted GPS coordinates.
    
    This predictor now provides both a default call (which uses a weighted average over the entire gallery)
    and a predict_topk function that returns the top-k predictions and their probabilities.
    """
    def __init__(self, model):
        self.model = model
        # Ensure the GPS gallery is on the same device as the model parameters.
        self.gps_gallery = self.model.gps_gallery.to(next(model.parameters()).device)
    
    def __call__(self, x):
        """
        Returns a single prediction per image computed as the softmax-weighted average over the GPS gallery.
        
        Args:
            x (torch.Tensor): Batch of image tensors of shape (B, 3, H, W).
            
        Returns:
            pred (torch.Tensor): Predicted GPS coordinates of shape (B, 2).
        """
        logits = self.model.forward(x, self.gps_gallery)  # shape: (B, num_gallery)
        probs = torch.softmax(logits, dim=1)               # shape: (B, num_gallery)
        pred = probs @ self.gps_gallery                    # shape: (B, 2)
        return pred

    def predict_topk(self, x, top_k):
        """
        Uses GeoCLIP's new predict_from_tensor function to obtain the top-k predictions and probabilities.
        Assumes that predict_from_tensor is updated to work in batch mode.
        
        Args:
            x (torch.Tensor): Batch of image tensors (B, 3, H, W).
            top_k (int): Number of top predictions to return.
            
        Returns:
            top_pred_gps (torch.Tensor): Tensor of shape (B, top_k, 2) with the top-k GPS coordinates.
            top_pred_prob (torch.Tensor): Tensor of shape (B, top_k) with the corresponding probabilities.
        """
        top_pred_gps, top_pred_prob = self.model.predict_from_tensor(x, top_k=top_k, apply_transforms=False)
        return top_pred_gps, top_pred_prob

class AttackGeoCLIP(RSAttack): # TODO: add an abstract attack class to all the attack later
    """
    A Sparse-RS attack class for GeoCLIP that uses a geodesic loss based on the top-k predictions.
    
    For untargeted attacks, our goal is to push the predicted GPS coordinates as far as possible
    from the ground-truth. In this implementation, the loss is computed per sample using the minimum
    haversine distance among the top-k predictions. We define the margin as:
    
        margin = 2500.0 - min_distance
    
    so that if the closest of the top-k predictions is farther than 2500 km, the margin becomes negative 
    (indicating total success), whereas if any prediction is closer than 1 km, the loss is high.
    
    The targeted attack formulation remains available for future adaptation.
    """
    def __init__(self, model, top_k=5, **kwargs):
        self.top_k = top_k
        # Wrap the GeoCLIP model using the updated predictor.
        predictor = GeoCLIPPredictor(model)
        super().__init__(predictor, **kwargs)
    
    def margin_and_loss(self, x, y):
        """
        Compute the margin and loss for a batch of perturbed images using top-k predictions.
        
        Args:
            x (torch.Tensor): Batch of perturbed images of shape (B, 3, H, W).
            y (torch.Tensor): Ground-truth GPS coordinates of shape (B, 2) (lat, lon in degrees).
            
        For untargeted attacks, the loss is defined as:
            margin = 2500.0 - min_distance
        where min_distance is the minimum haversine distance (in km) among the top-k predictions
        for each image.
        
        Returns:
            margin (torch.Tensor): Tensor of shape (B,) containing the margin for each sample.
            loss (torch.Tensor): Tensor of shape (B,) containing the loss for each sample (set equal to the margin).
        """
        # Obtain top-k predictions (B, top_k, 2) and their probabilities (B, top_k)
        top_pred_gps, top_pred_prob = self.predict.predict_topk(x, self.top_k)
        # Expand ground-truth coordinates to shape (B, top_k, 2)
        y_expanded = y.unsqueeze(1).expand_as(top_pred_gps)
        # Compute haversine distances for each top prediction; result is shape (B, top_k)
        distances = haversine_distance(top_pred_gps, y_expanded)
        # For each sample, take the minimum distance among the top-k predictions
        min_distance, _ = distances.min(dim=1)
        
        if not self.targeted:
            # For untargeted attacks: the objective is achieved if min_distance > 2500 km.
            # The margin (and loss) is high when min_distance is low.
            margin = 2500.0 - min_distance
            loss = margin
        else:
            # For targeted attacks, a different threshold can be used (e.g., self.tau_target)
            margin = distances - self.tau_target
            loss = margin
        
        return margin, loss
