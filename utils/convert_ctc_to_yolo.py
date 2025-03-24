from pathlib import Path
import cv2
import numpy as np
from tqdm import tqdm

data_dir = Path(__file__).parents[1] / "datasets"
target_size = (256, 32)  # (height, width) 

def convert_mask_to_boxes(mask: np.ndarray) -> list[tuple[float, float, float, float]]:
    boxes = []
    ids = np.unique(mask)
    ids = ids[ids != 0]

    for obj_id in ids:
        y, x = np.where(mask == obj_id)
        if len(x) == 0 or len(y) == 0:
            continue
        x_min, x_max = x.min(), x.max()
        y_min, y_max = y.min(), y.max()
        x_center = int((x_min + x_max) / 2)
        y_center = int((y_min + y_max) / 2)
        width = int(x_max - x_min)
        height = int(y_max - y_min)
        boxes.append((x_center, y_center, width, height))
    return boxes

def save_yolo_boxes(txt_path: Path, boxes: list[tuple[float, float, float, float]], shape: tuple[int, int], class_id: int = 0):
    h, w = shape
    with txt_path.open('w') as f:
        for x, y, bw, bh in boxes:
            f.write(f"{class_id} {x/w:.6f} {y/h:.6f} {bw/w:.6f} {bh/h:.6f}\n")

def convert_split(dataset_name: str, split_dir: str, target_size: tuple[int, int], data_dir: Path = data_dir):
    image_root = data_dir / dataset_name / "CTC" / split_dir
    out_img_dir = data_dir / dataset_name / "yolo" / "images" / split_dir
    out_lbl_dir = data_dir / dataset_name / "yolo" / "labels" / split_dir
    out_img_dir.mkdir(parents=True, exist_ok=True)
    out_lbl_dir.mkdir(parents=True, exist_ok=True)

    folders = sorted(image_root.glob("[0-9][0-9]"))
    for img_folder in tqdm(folders, desc=f"Processing {split_dir}", position=0):
        gt_folder = img_folder.parent / (img_folder.name + "_GT")

        if (gt_folder / 'SEG').exists():
            gt_folder = gt_folder / 'SEG'
        elif (gt_folder / 'TRA').exists():
            gt_folder = gt_folder / 'TRA'
        else:
            raise ValueError(f"No TRA or SEG folder found for {gt_folder.name}")

        image_files = sorted([f for f in img_folder.iterdir() if f.suffix == '.tif'])
        mask_files = sorted([f for f in gt_folder.iterdir() if f.suffix == '.tif'])

        for img_file, mask_file in tqdm(zip(image_files, mask_files), 
                                      desc=f"Folder {img_folder.name}", 
                                      position=1, 
                                      leave=False):
            
            image = cv2.imread(str(img_file))
            mask = cv2.imread(str(mask_file), cv2.IMREAD_UNCHANGED)
            if image is None or mask is None:
                print(f"Skipping {img_file.name} because it is None")
                continue

            # Calculate boxes from original mask
            boxes = convert_mask_to_boxes(mask)

            # Only resize the image, not the mask
            image = cv2.resize(image, (target_size[1], target_size[0]))

            base = f"{img_folder.name}{img_file.stem}"
            cv2.imwrite(str(out_img_dir / f"{base}.jpg"), image)
            # Use original mask.shape for normalization
            save_yolo_boxes(out_lbl_dir / f"{base}.txt", boxes, mask.shape)

if __name__ == "__main__":
    for split in ["train", "val", "test"]:
        convert_split(dataset_name="moma", split_dir=split, target_size=target_size)
