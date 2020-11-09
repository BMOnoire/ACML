from pathlib import Path

general = {
    "pickle_path": Path("pickels"),
    "imgs_path": Path("imgs")
}

nn = {
    "epochs": 3,
    "batch_size": 100,
    "optimizer": "Adam",
    "loss": "mean_squared_error",
    "metrics": ["accuracy"]
}


