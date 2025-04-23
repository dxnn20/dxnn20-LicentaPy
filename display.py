import os
import json
import torch

CONFIG = {
    "image_size": (460, 460),
    "device": torch.device('cuda' if torch.cuda.is_available() else 'cpu'),
    "data_dir": "E:\\dxnn20-LicentaPy\\archive\\train"
}

# Assuming the folders are like: /path/to/train/class_name/
data_dir = CONFIG["data_dir"].replace("train", "test")
class_names = sorted([d for d in os.listdir(data_dir) if os.path.isdir(os.path.join(data_dir, d))])

# Save mapping to JSON
class_id_to_name = {i: name for i, name in enumerate(class_names)}
with open("dermModelBackend/dermModel/src/main/resources/class_map.json", "w") as f:
    json.dump(class_id_to_name, f, indent=2)
