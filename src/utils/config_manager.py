"""JSON-based project configuration save / load."""
import json
import os


def default_config() -> dict:
    return {
        "experiments": [],
        "preprocessing": {
            "crop": {
                "enabled": False,
                "type": "manual",
                "x": 0,
                "y": 0,
                "w": 224,
                "h": 224,
                "center_size": 224,
            },
            "resize": {"width": 224, "height": 224},
            "normalize": {
                "use_imagenet": True,
                "mean": [0.485, 0.456, 0.406],
                "std": [0.229, 0.224, 0.225],
            },
            "augmentation": {
                "h_flip": {"enabled": True, "p": 0.5},
                "v_flip": {"enabled": False, "p": 0.5},
                "rotation": {"enabled": True, "degrees": 15},
                "color_jitter": {
                    "enabled": True,
                    "brightness": 0.2,
                    "contrast": 0.2,
                    "saturation": 0.2,
                    "hue": 0.05,
                },
                "gaussian_blur": {"enabled": False, "kernel_size": 5},
                "random_erasing": {"enabled": False, "p": 0.2},
                "random_resized_crop": {
                    "enabled": False,
                    "scale_min": 0.8,
                    "scale_max": 1.0,
                },
            },
        },
        "model": {
            "mode": "transfer",
            "transfer": {
                "architecture": "resnet50",
                "pretrained": True,
                "freeze_backbone": False,
                "unfreeze_last_n": 0,
                "dropout": 0.5,
            },
            "scratch": {
                "num_conv_blocks": 4,
                "base_filters": 32,
                "fc_layers": [256, 128],
                "batch_norm": True,
                "dropout": 0.5,
            },
        },
        "training": {
            "epochs": 50,
            "batch_size": {"value": "32", "values": "16,32,64", "use_grid": False},
            "learning_rate": {"value": "0.001", "values": "0.01,0.001,0.0001", "use_grid": False},
            "optimizer": {"value": "Adam", "values": "Adam,SGD,AdamW", "use_grid": False},
            "weight_decay": {"value": "1e-4", "values": "0,1e-4,1e-3", "use_grid": False},
            "loss": {"value": "MSE", "values": "MSE,MAE,Huber", "use_grid": False},
            "momentum": 0.9,
            "lr_scheduler": {
                "type": "StepLR",
                "step_size": 10,
                "gamma": 0.5,
                "t_max": 50,
                "patience": 5,
                "min_lr": 1e-6,
            },
            "early_stopping": {"enabled": True, "patience": 15, "min_delta": 1e-4},
            "grid_search_enabled": False,
            "output_dir": "./results",
            "num_workers": 0,
            "pin_memory": False,
        },
    }


def save(config: dict, path: str) -> None:
    os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(config, f, indent=2, ensure_ascii=False, default=str)


def load(path: str) -> dict:
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)
