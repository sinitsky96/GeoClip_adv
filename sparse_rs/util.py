import torch
import math

STREET_R = 1
CITY_R = 25
REGION_R = 200
COUNTRY_R = 750
CONTINENT_R = 2500


def haversine_distance(coord1, coord2):
    """
    Compute the Haversine distance (in kilometers) between two sets of (lat, lon) coordinates.
    Both coord1 and coord2 should be tensors of shape (..., 2) with lat, lon in degrees.
    This function is vectorized over any leading dimensions.
    """
    R = 6371.0  # Earth radius in kilometers
    
    # if type(coord1) is not tuple and type(coord2) is not tuple:
    #     coord2_d = coord2.clone().detach().to(coord1.device)
    # else:
    #     coord2_d = coord2

    coord2_d = coord2.clone().detach().to(coord1.device)
    # Convert degrees to radians.
    lat1 = coord1[..., 0] * math.pi / 180.0
    lon1 = coord1[..., 1] * math.pi / 180.0
    lat2 = coord2_d[..., 0] * math.pi / 180.0
    lon2 = coord2_d[..., 1] * math.pi / 180.0

    dlat = lat2 - lat1
    dlon = lon2 - lon1

    a = torch.sin(dlat / 2) ** 2 + torch.cos(lat1) * torch.cos(lat2) * torch.sin(dlon / 2) ** 2
    c = 2 * torch.atan2(torch.sqrt(a), torch.sqrt(1 - a))
    distance = R * c
    return distance