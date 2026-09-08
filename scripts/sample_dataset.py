import os
import random
import shutil

def sample_split(split, target_count):
    source_img = f"data/raw/auto_dataset_full/{split}/images"
    source_lbl = f"data/raw/auto_dataset_full/{split}/labels"
    dest_img = f"data/auto_dataset/{split}/images"
    dest_lbl = f"data/auto_dataset/{split}/labels"

    os.makedirs(dest_img, exist_ok=True)
    os.makedirs(dest_lbl, exist_ok=True)

    all_images = os.listdir(source_img)
    sample_size = min(target_count, len(all_images))
    selected = random.sample(all_images, sample_size)

    for img_name in selected:
        shutil.copy(f"{source_img}/{img_name}", f"{dest_img}/{img_name}")
        label_name = os.path.splitext(img_name)[0] + ".txt"
        label_path = f"{source_lbl}/{label_name}"
        if os.path.exists(label_path):
            shutil.copy(label_path, f"{dest_lbl}/{label_name}")

    print(f"✅ {split}: sampled {sample_size} images")

sample_split("train", 1200)
sample_split("valid", 300)
sample_split("test", 100)