import os
import cv2
import numpy as np
from tqdm import tqdm

TILE_SIZE = 512
OVERLAP = 100
DATASET_PATH = "dead-bird-detection\exports"
OUTPUT_PATH = "PIC_2_test_tiled_512_v2"

#EMPTY_LABEL = "0 0 0 0 0"  # class_id = 0 pour les tuiles sans objets

# renvoie la liste de tuples (class_id, x_center, y_center, width, height)=> normalisés
def load_yolo_labels(label_path):
    """Charge les labels normalisés."""
    if not os.path.exists(label_path):
        return []
    labels = []
    with open(label_path, "r") as f:
        for line in f.readlines():
            parts = line.strip().split()
            class_id = int(parts[0])
            x_center, y_center, width, height = map(float, parts[1:])
            labels.append((class_id, x_center, y_center, width, height))
    return labels


def save_yolo_labels(path, labels):
    """Enregistre les labels dans un fichier."""
    with open(path, "w") as f:
        for label in labels:
            f.write(" ".join(map(str, label)) + "\n")


def pad_tile(tile):
    """Pad les tuiles incomplètes pour qu'elles deviennent 512×512."""
    h, w, _ = tile.shape
    padded = np.zeros((TILE_SIZE, TILE_SIZE, 3), dtype=np.uint8)
    padded[:h, :w] = tile
    return padded


def split_image_and_labels(image_path, label_path, output_img_folder, output_lbl_folder):
    img = cv2.imread(image_path)
    if img is None:
        return

    h, w, _ = img.shape
    labels = load_yolo_labels(label_path)
    step = TILE_SIZE - OVERLAP

    base_name = os.path.splitext(os.path.basename(image_path))[0]

    for y in range(0, h, step):
        for x in range(0, w, step):

            # Extraire la tuile
            tile = img[y:y + TILE_SIZE, x:x + TILE_SIZE]
            tile_h, tile_w, _ = tile.shape

            # Vérifier si la tuile est "trop petite" pour être utile
            if tile_h < 0.5 * TILE_SIZE or tile_w < 0.5 * TILE_SIZE:
                continue  # Ignorer les tuiles vraiment trop petites

        # Pad seulement si tuile correcte mais bordure incomplète
            if tile_h != TILE_SIZE or tile_w != TILE_SIZE:
                tile = pad_tile(tile)

            tile_filename = f"{base_name}_{x}_{y}.jpg"
            tile_path = os.path.join(output_img_folder, tile_filename)

            new_labels = []

            # Recalcul des labels pour la tuile
            for class_id, xc, yc, w_norm, h_norm in labels:
                x_abs = xc * w
                y_abs = yc * h
                box_w = w_norm * w
                box_h = h_norm * h

                # centre dans la tuile ?
                if x <= x_abs <= x + TILE_SIZE and y <= y_abs <= y + TILE_SIZE:
                    new_xc = (x_abs - x) / TILE_SIZE
                    new_yc = (y_abs - y) / TILE_SIZE
                    new_w = box_w / TILE_SIZE
                    new_h = box_h / TILE_SIZE
                    new_labels.append((0, new_xc, new_yc, new_w, new_h))

            # Sauvegarde de la tuile
            cv2.imwrite(tile_path, tile)

            # Sauvegarde des labels
            label_filename = f"{base_name}_{x}_{y}.txt"
            out_label_path = os.path.join(output_lbl_folder, label_filename)

            if new_labels:
                save_yolo_labels(out_label_path, new_labels)
            else:
                # Label vide (classe 0)
                #with open(out_label_path, "w") as f:
                #    f.write(EMPTY_LABEL + "\n")
                open(out_label_path, "w").close()

for split in ["train","valid","test"]:
    image_dir = os.path.join(DATASET_PATH, split, "images")
    label_dir = os.path.join(DATASET_PATH, split, "labels")

    out_img = os.path.join(OUTPUT_PATH, split, "images")
    out_lbl = os.path.join(OUTPUT_PATH, split, "labels")

    os.makedirs(out_img, exist_ok=True)
    os.makedirs(out_lbl, exist_ok=True)

    if not os.path.exists(image_dir):
        print(f"Erreur : dossier introuvable : {image_dir}")
        continue

    image_files = [f for f in os.listdir(image_dir) if f.endswith((".jpg", ".png"))]

    for image_file in tqdm(image_files, desc=f"Tiling ({split})"):
        img_path = os.path.join(image_dir, image_file)
        lbl_path = os.path.join(label_dir, os.path.splitext(image_file)[0] + ".txt")
        split_image_and_labels(img_path, lbl_path, out_img, out_lbl)
