from pathlib import Path

general = {
    "pickle_path": Path("pickels"),
    "pickle_data_path": Path("pickels")
}

nn = {
    "epochs": 3,
    "batch_size": 250,
    "optimizer": "Adam",
    "loss": "mean_squared_error",
    "metrics": ["accuracy"]
}


