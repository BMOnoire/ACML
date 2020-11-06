from pathlib import Path

general = {
    "pickle_path": Path("pickels"),
    "pickle_data_path": Path("pickels")
}

nn = {
    "epochs": 10,
    "optimizer": "Adam",
    "loss": "categorical_crossentropy",
    "metrics": ["accuracy"]
}


