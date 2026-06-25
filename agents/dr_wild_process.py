import os
import cv2
import numpy as np
from PIL import Image
from tqdm import tqdm

if __name__ == "__main__":
    scene_name = "TengWangGe"
    height, width = 989, 1320
    in_path = "/data/guangyu/aLit/record/diffusionrenderer/Wild/{}".format(scene_name)
    group_items = sorted(os.listdir(in_path))
    
    ot_alb_path = "/data/guangyu/aLit/record/diffusionrenderer/results/{}/albedo".format(scene_name)
    os.makedirs(ot_alb_path, exist_ok=True)
    ot_ird_path = "/data/guangyu/aLit/record/diffusionrenderer/results/{}/irradiance".format(scene_name)
    os.makedirs(ot_ird_path, exist_ok=True)
    
    cnt = 1
    for group_name in tqdm(group_items):
        in_group_path = os.path.join(in_path, group_name)
        img_name = os.path.join(in_group_path, "0000.0000.basecolor.jpg")
        img = Image.open(img_name).resize((width, height))
        img.save(os.path.join(ot_alb_path, "{}.jpg".format(group_name)))
        img.save(os.path.join(ot_ird_path, "{}.jpg".format(group_name)))
    
    
    