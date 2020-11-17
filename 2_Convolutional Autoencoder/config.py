from pathlib import Path

general = {
    "pickle_path": Path("pickels"),
    "imgs_path": Path("imgs"),
    "models_path": Path("models"),
    "images_list": [0, 2, 3, 5, 6]
}

nn = {
    "epochs": 50,
    "batch_size": 50,
    "optimizer": "Adam",
    "loss": "mean_squared_error",
    "metrics": ["accuracy"]
}


