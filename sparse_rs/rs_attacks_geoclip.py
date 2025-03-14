import torch
import torch.nn.functional as F
import math
from sparse_rs.rs_attacks import RSAttack  # reuse the existing Sparse-RS attack implementation

def haversine_distance(coord1, coord2):
    """
    Computes the Haversine distance (in kilometers) between two sets of (lat, lon) coordinates.
    Both coord1 and coord2 should be tensors of shape (batch, 2) with latitude and longitude in degrees.
    """
    R = 6371.0  # Earth radius in kilometers
    # Convert degrees to radians
    lat1 = coord1[:, 0] * math.pi / 180.0
    lon1 = coord1[:, 1] * math.pi / 180.0
    lat2 = coord2[:, 0] * math.pi / 180.0
    lon2 = coord2[:, 1] * math.pi / 180.0

    dlat = lat2 - lat1
    dlon = lon2 - lon1

    a = torch.sin(dlat / 2) ** 2 + torch.cos(lat1) * torch.cos(lat2) * torch.sin(dlon / 2) ** 2
    c = 2 * torch.atan2(torch.sqrt(a), torch.sqrt(1 - a))
    distance = R * c
    return distance

class GeoCLIPPredictor():
    """
    Wraps a GeoCLIP model so that calling it with a batch of images returns predicted GPS coordinates.
    The prediction is computed as the softmax–weighted average over the GPS gallery.
    """
    def __init__(self, model):
        self.model = model
        # Ensure the gps_gallery is on the same device as the model parameters.
        self.gps_gallery = self.model.gps_gallery.to(next(model.parameters()).device)
    
    def __call__(self, x):
        # Run GeoCLIP forward: x is of shape (batch, 3, H, W)
        logits = self.model(x, self.gps_gallery)  # logits shape: (batch, num_gallery)
        probs = torch.softmax(logits, dim=1)
        # Compute the predicted coordinate as the weighted average of the gallery
        pred = probs @ self.gps_gallery  # resulting shape: (batch, 2)
        return pred

class RSAttackGeoCLIP(RSAttack):
    """
    A Sparse-RS attack class for GeoCLIP that uses a geodesic loss.
    
    For an untargeted attack, we want to maximize the distance between the prediction and the ground-truth.
    We define the margin as:
    
        margin = 2500.0 - distance
    
    so that predictions with distance > 2500 km (total success) yield a negative margin, while those closer than 1 km are highly penalized.
    
    The targeted case is left unchanged (or can be adapted separately).
    """
    def __init__(self, model, **kwargs):
        # Wrap the GeoCLIP model in our predictor.
        predictor = GeoCLIPPredictor(model)
        super().__init__(predictor, **kwargs)
    
    def margin_and_loss(self, x, y):
        """
        Computes the margin and loss for a batch of perturbed images.
        
        Parameters:
          x: Batch of images (perturbed) [batch, 3, H, W]
          y: Ground-truth GPS coordinates as a tensor of shape (batch, 2) with lat and lon in degrees.
        
        For untargeted attacks, the loss is defined as:
        
          margin = 2500.0 - distance
        
        so that if the predicted distance exceeds 2500 km the margin becomes negative (attack succeeds),
        and if it is below 1 km (or any low value) the margin is very high (attack fails).
        """
        # Obtain predicted GPS coordinates
        pred = self.predict(x)  # shape: (batch, 2)
        # Compute geodesic distance in km between prediction and ground-truth
        distance = haversine_distance(pred, y)
        if not self.targeted:
            # For untargeted attacks: maximize distance from y
            margin = 2500.0 - distance  # attack is successful if distance > 2500 km
            loss = margin
        else:
            # For targeted attacks, one could define a different threshold (e.g. tau_target)
            # Here we simply keep the original formulation:
            margin = distance - self.tau_target
            loss = margin
        return margin, loss
