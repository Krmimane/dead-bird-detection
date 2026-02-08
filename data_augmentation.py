import os
import cv2
import albumentations as A
import numpy as np

# ============================
#   TRANSFORMATIONS
# ============================

transform = A.Compose(
    [
        A.HorizontalFlip(p=0.5),
        A.VerticalFlip(p=0.3),

        A.RandomBrightnessContrast(p=0.4),

        A.GaussianBlur(blur_limit=5, p=0.3),

        A.GaussNoise(
            std_range=(0.1, 0.2),
            mean_range=(0, 0),
            per_channel=True,
            noise_scale_factor=1,
            p=0.4
        ),

        A.MultiplicativeNoise(multiplier=(0.9, 1.1), p=0.3),

        A.CoarseDropout(
            num_holes_range=(1, 5),
            hole_height_range=(0.1, 0.2),
            hole_width_range=(0.1, 0.2),
            fill=0,
            p=0.4
        ),

        A.Affine(
            translate_percent=0.03,
            scale=1.0,              # taille inchangée
            rotate=(-5, 5),
            mode=cv2.BORDER_REFLECT_101,
            p=0.5
        ),
    ],
    bbox_params=A.BboxParams(
        format="yolo",
        label_fields=["class_labels"],
        min_visibility=0.3,
        clip=True                # empêche bbox > 1
    )
)

# ============================
#   PARAMÈTRES
# ============================

DATASET_ROOT = "C:\\Users\\DELL\\Documents\\S9\\Projet_Pic\\PIC_2_test_tiled_512_v2"
OUTPUT_ROOT = "dead-bird-detection\\augmented_dataset"

N_AUG_PER_IMAGE = 4
EXPECTED_W, EXPECTED_H = 512, 512

splits = ["test","valid"]  

# ============================
#   LECTURE LABELS YOLO
# ============================

def load_yolo_labels(path):
    boxes, classes = [], []
    with open(path, "r") as f:
        for line in f:
            cls, xc, yc, w, h = map(float, line.split())
            if int(cls) == 0:          #  seulement classe 0
                boxes.append([xc, yc, w, h])
                classes.append(0)
    return boxes, classes


# ============================
#   AUGMENTATION
# ============================

for split in splits:

    img_dir = os.path.join(DATASET_ROOT, split, "images")
    lbl_dir = os.path.join(DATASET_ROOT, split, "labels")

    out_img_dir = os.path.join(OUTPUT_ROOT, split, "images")
    out_lbl_dir = os.path.join(OUTPUT_ROOT, split, "labels")

    os.makedirs(out_img_dir, exist_ok=True)
    os.makedirs(out_lbl_dir, exist_ok=True)

    print(f"\n--- Processing {split} ---")

    for fname in os.listdir(img_dir):

        if not fname.lower().endswith((".jpg", ".png")):
            continue

        img_path = os.path.join(img_dir, fname)
        lbl_path = os.path.join(lbl_dir, fname.rsplit(".", 1)[0] + ".txt")

        if not os.path.exists(lbl_path):
            continue

        img = cv2.imread(img_path)
        h, w = img.shape[:2]

        if (w, h) != (EXPECTED_W, EXPECTED_H):
            continue

        boxes, classes = load_yolo_labels(lbl_path)
        if len(boxes) == 0:
            continue

        for i in range(N_AUG_PER_IMAGE):

            augmented = transform(
                image=img,
                bboxes=boxes,
                class_labels=classes
            )

            aug_img = augmented["image"]
            aug_boxes = augmented["bboxes"]

            if len(aug_boxes) == 0:
                continue

            #  mêmes noms que l’original
            name, ext = fname.rsplit(".", 1)
            out_img_path = os.path.join(out_img_dir, f"{name}_aug{i}.{ext}")
            out_lbl_path = os.path.join(out_lbl_dir, f"{name}_aug{i}.txt")

            cv2.imwrite(out_img_path, aug_img)

            with open(out_lbl_path, "w") as f:
                for xc, yc, bw, bh in aug_boxes:
                    f.write(f"0 {xc:.6f} {yc:.6f} {bw:.6f} {bh:.6f}\n")

    print(f" Fin split {split}")

print("\n====================")
print(" AUGMENTATION TERMINÉE")
print("====================")
