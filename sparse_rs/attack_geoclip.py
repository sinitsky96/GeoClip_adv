import torch
from rs_attacks import RSAttack
from sparse_rs.util import haversine_distance, CONTINENT_R, STREET_R

class AttackGeoCLIP(RSAttack): # TODO: add an abstract attack class to all the attacks we use later
    """
    A Sparse-RS attack class for GeoCLIP that uses a geodesic loss based on the top-k predictions.
    
    For untargeted attacks, our goal is to push the predicted GPS coordinates as far as possible
    from the ground-truth. In this implementation, the loss is computed per sample using the minimum
    haversine distance among the top-k predictions. We define the margin as:
    
        margin = 2500.0 - min_distance
    
    so that if the closest of the top-k predictions is farther than 2500 km, the margin becomes negative 
    (indicating total success), whereas if any prediction is closer than 1 km, the loss is high.
    """
    def __init__(self, model, **kwargs):
        self.model = model
        super().__init__(self.predict, **kwargs)

    def predict(self, x):
        output, _ = self.model.predict_from_tensor(x)
        return output


    def margin_and_loss(self, x, y):
        """
        Adapted margin_and_loss for GeoCLIP adversarial attack based on distance.
        
        Args:
            x (torch.Tensor): Batch of images (B, 3, H, W)
            y (torch.Tensor): Ground-truth GPS coordinates (B, 2)
        """
        predicted_gps, _ = self.model.predict_from_tensor(x, top_k=1, apply_transforms=False)
        # predicted_gps = predicted_gps.to(y.device)
        # print(f"predicted_gps : {predicted_gps}")

        # Compute the haversine distance between predicted GPS and ground-truth y.
        distance = haversine_distance(predicted_gps, y)  # shape: (B,)
        
        if not self.targeted:
            margin = torch.sub(CONTINENT_R, distance)
            loss = -distance
        
        # For targeted attacks, the distance to be less than a target threshold (STREET_R).
        else:
            margin = torch.sub(distance, STREET_R)
            loss = distance
        
        return margin, loss