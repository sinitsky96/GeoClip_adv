import torch
from sparse_rs.rs_attacks import RSAttack
from sparse_rs.util import haversine_distance, CONTINENT_R, STREET_R
from transformers import CLIPProcessor, CLIPFeatureExtractor, CLIPTokenizer
import torch.nn.functional as F
import os
from data.Im2GPS3k.download import load_places365_categories

def find_nearest_neighbor_index(gps_gallery, coord):
    distances = haversine_distance(gps_gallery, coord.unsqueeze(0), True)  # shape: (N,)
    nn_index = torch.argmin(distances).item()
    return nn_index

def coords_to_class_indices_nn(gps_gallery, coords):
    # coords: (B, 2)
    label_indices = []
    for i in range(coords.shape[0]):
        index = find_nearest_neighbor_index(gps_gallery, coords[i])
        label_indices.append(index)
    return torch.LongTensor(label_indices)


class AttackGeoCLIP(RSAttack):
    """
    A Sparse-RS attack class for GeoCLIP that uses a geodesic loss based on the top-k predictions.
    
    For untargeted attacks, our goal is to push the predicted GPS coordinates as far as possible
    from the ground-truth. In this implementation, the loss is computed per sample using the minimum
    haversine distance among the top-k predictions. We define the margin as:
    
        margin = 2500.0 - min_distance
    
    so that if the closest of the top-k predictions is farther than 2500 km, the margin becomes negative 
    (indicating total success), whereas if any prediction is closer than 1 km, the loss is high.
    """
    def __init__(self, model, y_test, **kwargs):
        self.model = model
        super().__init__(self.predict, **kwargs)
        gps_gallery = self.model.gps_gallery.to(self.device)
        self.label_indices = coords_to_class_indices_nn(gps_gallery, y_test).long().to(self.device)
        self.label_dict = {}
        for i in range(y_test.shape[0]):
            # Convert coordinate tensor to tuple (e.g., (lat, lon))
            coord_tuple = tuple(y_test[i].tolist())
            self.label_dict[coord_tuple] = self.label_indices[i]

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
        # print("in margin loss")
        # print(f"gallery device: {self.model.gps_gallery.device}")
        # print(y)
        logits = self.model.predict_logits(x)
        probs_per_image = logits.softmax(dim=-1)
        top_pred = torch.topk(probs_per_image, 1, dim=1) # for gallery query
        gps_gallery = self.model.gps_gallery.to(self.device)
        # print(f"model device: {self.model.device}")
        # print(f"gallery device: {self.model.gps_gallery.device}, indecies device: {top_pred.indices.device}")
        top_pred_gps = gps_gallery[top_pred.indices]
        predicted_gps = top_pred_gps.squeeze(1) # removes the singleton top_k (we use k=1) dimension: [100, 1, 2] -> [100, 2]

        distance = haversine_distance(predicted_gps, y, True)  # shape: (B,)
        # print(y)

        if not self.targeted:
            margin = torch.sub(CONTINENT_R, distance)
            if self.loss == 'ce':
                label_indices = torch.LongTensor(
                    [self.label_dict[tuple(coord.tolist())] for coord in y]
                ).to(self.device)
                xent = F.cross_entropy(logits, label_indices, reduction='none')
                loss = -1.0 * xent
            elif self.loss == 'margin':
                loss = - margin
        else:
            margin = torch.sub(distance, STREET_R)
            if self.loss == 'ce':
                label_indices = torch.LongTensor(
                    [self.label_dict[tuple(coord.tolist())] for coord in y]
                ).to(self.device)
                xent = F.cross_entropy(logits, label_indices, reduction='none')
                loss = xent
            elif self.loss == 'margin':
                loss = margin
        
        return margin, loss
    

class ClipWrap():
    def __init__(self, model, data_path, device, enable_grad=False):

        # self.feature_extractor = CLIPFeatureExtractor.from_pretrained(
        #     "openai/clip-vit-large-patch14",
        #     do_resize=False, do_center_crop=False, do_normalize=False
        # )
        # self.tokenizer = CLIPTokenizer.from_pretrained("openai/clip-vit-large-patch14")
        # self.processor = CLIPProcessor(self.feature_extractor, self.tokenizer)
        self.processor = CLIPProcessor.from_pretrained("openai/clip-vit-large-patch14",
                                                       do_rescale=False,
                                                       do_resize=False,
                                                       do_center_crop=False,
                                                       do_normalize=False
                                                       )
        self.prompts = load_places365_categories(os.path.join(data_path, 'places365_cat.txt'))
        self.device = device
        self.model = model
        self.enable_grad = enable_grad

    def __call__(self, x):
        return self.get_logits(x)
    
    def named_modules(self):
        return self.model.named_modules()

    def get_logits(self, x):
        # print(self.prompts)
        inputs = self.processor(images=x,               
                                text=self.prompts,       
                                return_tensors="pt",
                                padding=True,
                                ).to(self.device)
        if self.enable_grad:
            outputs = self.model(**inputs)
        else:
            with torch.no_grad():
                outputs = self.model(**inputs)

        return outputs.logits_per_image

    
class AttackCLIP(RSAttack):
    def __init__(self, model, data_path, **kwargs):
        # self.processor = CLIPProcessor.from_pretrained("openai/clip-vit-large-patch14")
        # self.prompts = load_places365_categories(os.path.join(data_path, 'places365_cat.txt'))
        self.model = model
        super().__init__(model, **kwargs)


    # def get_logits(self, x):
    #     inputs = self.processor(images=x,               
    #                             text=self.prompts,       
    #                             return_tensors="pt",
    #                             padding=True)

    #     with torch.no_grad():
    #         outputs = self.model(**inputs)

    #     return outputs.logits_per_image

    # def predict(self, x):
    #     logits = self.get_logits(x)
    #     probs = logits.softmax(dim=1)
    #     predictions = probs.argmax(dim=1)
    #     return predictions # class ids

    
    def margin_and_loss(self, x, y):
        """
        Adapted margin_and_loss for GeoCLIP adversarial attack based on distance.
        
        Args:
            x (torch.Tensor): Batch of images (B, 3, H, W)
            y (torch.Tensor): Ground-truth GPS coordinates (B, 2)
        """
        logits = self.model(x)

        # same as in the attack_rs.py
        xent = F.cross_entropy(logits, y, reduction='none')

        u = torch.arange(len(x), device=self.device)
        y_corr = logits[u, y].clone()
        logits[u, y] = -float('inf')
        y_others = logits.max(dim=-1)[0]

        if not self.targeted:
            if self.loss == 'ce':
                return y_corr - y_others, -1. * xent
            elif self.loss == 'margin':
                return y_corr - y_others, y_corr - y_others
        else:
            # targeted
            return y_others - y_corr, xent