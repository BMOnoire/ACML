

EPOCHS = 5000
# show_n_time shows how many time you can see the car during the EPOCHS

TEST_LIST = [
    {
        "id": "first",
        "epochs": EPOCHS,
        "show_n_time": 5,
        "q_table_dimension": 20,
        "learning_rate": 0.1,
        "discount": 0.95,
        "epsilon": 0.5,
        "epsilon_decaying_range": (1, EPOCHS//2)
    },
    {
        "id": "custom",
        "epochs": EPOCHS,
        "show_n_time": 5,
        "q_table_dimension": 40,
        "learning_rate": 0.25,
        "discount": 0.95,
        "epsilon": 0.5,
        "epsilon_decaying_range": (1, EPOCHS//2)
    }
]


