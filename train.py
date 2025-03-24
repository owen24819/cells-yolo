from ultralytics import YOLO
import torch
from omegaconf import OmegaConf

def main():
    # Load config and merge with CLI arguments
    cfg = OmegaConf.load('cfg.yaml')
    cli_cfg = OmegaConf.from_cli()
    cfg = OmegaConf.merge(cfg, cli_cfg)

    model = YOLO(f"{cfg.model}.pt")
    device = torch.device('cpu')
    model = model.to(device)

    model.info()

    model.train(
        data=cfg.data,  # Path to data config
        imgsz=cfg.imgsz,
        epochs=cfg.epochs,
        batch=cfg.batch,
        project=cfg.project,
        name=cfg.name,
        exist_ok=True
    )

if __name__ == "__main__":
    main()
