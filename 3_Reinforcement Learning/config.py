from pathlib import Path

EPOCHS = 1000

TEST_LIST = [
    {
        "id": "first",
        "epochs": EPOCHS,
        "show_n_time": 0,
        "q_table_dimension": 10,
        "learning_rate": 0.1,
        "discount": 0.95,
        "epsilon": 0.5,
        "epsilon_decaying_range": (1, EPOCHS//2)
    },
    {
        "id": "second",
        "epochs": EPOCHS,
        "show_n_time": 0,
        "q_table_dimension": 20,
        "learning_rate": 0.1,
        "discount": 0.95,
        "epsilon": 0.5,
        "epsilon_decaying_range": (1, EPOCHS//2)
    },
    {
        "id": "third",
        "epochs": EPOCHS,
        "show_n_time": 0,
        "q_table_dimension": 100,
        "learning_rate": 0.1,
        "discount": 0.95,
        "epsilon": 0.5,
        "epsilon_decaying_range": (1, EPOCHS//2)
    }
]

