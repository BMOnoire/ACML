from pathlib import Path

general = {
    "pickle_path": Path("pickels"),
    "imgs_path": Path("imgs"),
    "models_path": Path("models")
}

nn = {
    "epochs": 500,
    "batch_size": 50,
    "optimizer": "Adam",
    "loss": "mean_squared_error",
    "metrics": ["accuracy"]
}


