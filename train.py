# Third party imports
from ultralytics import YOLO

# Local imports
import utils

def main():

    # Get the configuration - Use resume=True and args-path=path/to/args.yaml to resume training
    cfg = utils.get_config()

    # Download the dataset if not already downloaded
    utils.download_dataset(cfg)
    
    # Load the model
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
