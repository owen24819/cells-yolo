from pathlib import Path

from omegaconf import OmegaConf
from ultralytics import YOLO

from utils import download_dataset

def main():
    # Load config and merge with CLI arguments
    cli_cfg = OmegaConf.from_cli()

    # If resume training, load the original config file, otherwise load the default config file
    if 'resume' in cli_cfg and cli_cfg.resume and 'model' in cli_cfg and cli_cfg.model:
        cfg = OmegaConf.load(f"{Path(cli_cfg.model).parents[1]}/args.yaml")
    else:
        cfg = OmegaConf.load('cfg.yaml')

    cfg = OmegaConf.merge(cfg, cli_cfg)

    download_dataset(cfg)
    
    model = YOLO(cfg.model)
    model.info()

    model.train(
        data=cfg.data,  # Use absolute path to data config
        imgsz=cfg.imgsz,
        epochs=cfg.epochs,
        batch=cfg.batch,
        project=cfg.project,
        name=cfg.name,
        exist_ok=True,
        resume=cfg.resume,
    )

if __name__ == "__main__":
    main()
