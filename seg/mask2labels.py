import os
import cv2
import numpy as np
from skimage import measure

mask_dir = r"C:\Users\524ha\Desktop\Resources\BasketballGameTracker\data\seg\courtv2-Final\valid\masks"
out_dir = r"C:\Users\524ha\Desktop\Resources\BasketballGameTracker\seglab\labels"
os.makedirs(out_dir, exist_ok=True)

for fname in os.listdir(mask_dir):
    mask = cv2.imread(os.path.join(mask_dir, fname), cv2.IMREAD_UNCHANGED)
    h, w = mask.shape[:2]
    label_lines = []
    
    for class_id in np.unique(mask):
        if class_id == 0:
            continue  # arka planı atla
        # class_id'ye ait piksellerin konturlarını bul
        contours, _ = cv2.findContours((mask == class_id).astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        for contour in contours:
            if len(contour) < 3:
                continue
            # normalize et
            points = contour.squeeze().astype(float)
            points[:, 0] /= w
            points[:, 1] /= h
            coords = " ".join([f"{x:.6f} {y:.6f}" for x, y in points])
            label_lines.append(f"{int(class_id)} {coords}")
    
    out_name = os.path.splitext(fname)[0] + ".txt"
    with open(os.path.join(out_dir, out_name), "w") as f:
        f.write("\n".join(label_lines))
