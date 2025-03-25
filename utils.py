import zipfile
from pathlib import Path

import cv2
import numpy as np
import requests
from omegaconf import OmegaConf
from tqdm import tqdm

def convert_mask_to_boxes(mask: np.ndarray) -> list[tuple[float, float, float, float]]:
    """
    Convert a mask to a list of bounding boxes.

    Args:
        mask (np.ndarray): The mask to convert
    Returns:    
        boxes (list[tuple[float, float, float, float]]): The list of bounding boxes
    """

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
    """
    Save the bounding boxes to a YOLO format text file.

    Args:
        txt_path (Path): The path to save the text file
        boxes (list[tuple[float, float, float, float]]): The list of bounding boxes
        shape (tuple[int, int]): The shape of the image
        class_id (int): The class ID
    """

    h, w = shape
    with txt_path.open('w') as f:
        for x, y, bw, bh in boxes:
            f.write(f"{class_id} {x/w:.6f} {y/h:.6f} {bw/w:.6f} {bh/h:.6f}\n")

def convert_split(dataset_name: str, split_dir: str, target_size: tuple[int, int], data_dir: Path):
    """
    Convert a CTC dataset split to YOLO format.

    Args:
        dataset_name (str): Name of the dataset
        split_dir (str): Directory containing the split data
        target_size (tuple[int, int]): Target size for the images
        data_dir (Path): Directory containing the dataset
    """

    image_root = data_dir / dataset_name / "CTC" / split_dir
    out_img_dir = data_dir / dataset_name / "yolo" / "images" / split_dir
    out_lbl_dir = data_dir / dataset_name / "yolo" / "labels" / split_dir

    if out_img_dir.exists() and out_lbl_dir.exists():
        print(f"Dataset {dataset_name} {split_dir} already exists, skipping...")
        return
    
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

def convert_ctc_to_yolo(dataset_name: str, target_size: tuple[int, int], data_dir: Path):
    for split in ["train", "val", "test"]:
        convert_split(dataset_name=dataset_name, split_dir=split, target_size=target_size, data_dir=data_dir)

def download(url: str, dir: Path, unzip: bool = False, delete: bool = False):
    """
    Download a file from a URL and optionally extract it.
    
    Args:
        url (str): URL to download from
        dir (Path): Directory to save the file to
        unzip (bool): Whether to unzip the downloaded file
        delete (bool): Whether to delete the zip file after extraction
    """

    
    dir.mkdir(parents=True, exist_ok=True)
    
    # Get filename from URL
    filename = url.split('?')[0].split('/')[-1]
    filepath = dir / filename
    
    if filepath.exists():
        print(f"File {filename} already exists, skipping download...")
    else:    
        # Download with progress bar
        response = requests.get(url, stream=True)
        total_size = int(response.headers.get('content-length', 0))
        
        with open(filepath, 'wb') as f, tqdm(
            desc=f"Downloading {filename}",
            total=total_size,
            unit='iB',
            unit_scale=True,
            unit_divisor=1024,
        ) as pbar:
            for data in response.iter_content(chunk_size=1024):
                size = f.write(data)
            pbar.update(size)
    
    if unzip and filepath.suffix == '.zip':
        with zipfile.ZipFile(filepath, 'r') as zip_ref:
            zip_ref.extractall(dir)
        if delete:
            filepath.unlink()

def download_moma_dataset(dir: Path):
    """
    Download MOMA dataset from Zenodo and extract it to the specified directory.
    
    Args:
        dir (str | Path): Directory to download and extract the dataset to
    """
    # dir = Path(dir)
    if (dir / 'CTC').exists():
        print("CTC folder already exists, skipping download...")
        return

    # Download dataset
    print("Downloading CTC.zip from Zenodo...")
    url = "https://zenodo.org/records/11237127/files/CTC.zip?download=1"
    download(url=url, dir=dir, unzip=True, delete=True)
    
    print(f"Download complete! Dataset saved in {dir}")

def download_ctc_dataset(dataset_name: str, data_dir: Path):
    """
    Download CTC dataset from Zenodo and extract it to the specified directory.
    
    Args:
        dataset_name (str): Name of the dataset to download
        data_dir (Path): Directory to download and extract the dataset to
    """
    if dataset_name == "moma":
        download_moma_dataset(data_dir / dataset_name)
    else:
        raise ValueError(f"Dataset {dataset_name} not supported")

def download_dataset(cfg: OmegaConf):
    """
    Download and convert a CTC dataset to YOLO format.

    Args:
        cfg (OmegaConf): Configuration object containing dataset details
    """

    data_dir = Path(__file__).parent / "datasets"

    if cfg.dataset == "moma":
        target_size = (256, 32)
    else:
        raise ValueError(f"Dataset {cfg.dataset_name} not supported")
    
    download_ctc_dataset(dataset_name=cfg.dataset, data_dir=data_dir)

    convert_ctc_to_yolo(dataset_name=cfg.dataset, target_size=target_size, data_dir=data_dir)

