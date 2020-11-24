import os
import gym
import numpy as np
import matplotlib.pyplot as plt
import time
#os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2' # lighter system log
import config as cfg
import janitor as jn


# ACTIONS
GO_LEFT, STAY, GO_RIGHT = 0, 1, 2


EPOCHS = 50

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

date = time.strftime("%Y_%m_%d_%H_%M_%S")

def plot_graph_result(test_name, epoch_list, avg_list, max_list, min_list, show=False):
    plt.plot(epoch_list, avg_list, label="avg", color="green")
    plt.plot(epoch_list, max_list, label="max", color="red")
    plt.plot(epoch_list, min_list, label="min", color="blue")
    plt.legend(loc='upper left')

    date = time.strftime("%Y_%m_%d_%H_%M_%S")
    plt.savefig(f"imgs\\graph_test_{test_name}_{date}")

    if show:
        plt.show()


def plot_heat_map(test_name, q_table, show=False):
    heatmap = np.max(q_table, 2)
    plt.imshow(heatmap, cmap='YlOrRd', interpolation='nearest')
    plt.title("State Value function")
    plt.xlabel("Speed (-0.07 to 0.07)")
    plt.ylabel("Position (-1.2 to 0.6)")
    plt.gca().invert_yaxis()
    plt.colorbar()

    date = time.strftime("%Y_%m_%d_%H_%M_%S")
    plt.savefig(f"imgs\\heatmap_test_{test_name}_{date}")

    if show:
        plt.show()


def launch_new_q_learning_test(test_id, epoch_number, n_show, q_table_dimension, learning_rate, discount, epsilon, epsilon_decaying_range):
    epoch_rewards = []
    aggr_ep_rewards = {
        "epoch": [],
        "avg_value": [],
        "min_value": [],
        "max_value": []
    }

    np.random.seed(1)
    env = gym.make("MountainCar-v0")

    observation_size = len(env.observation_space.high) # S, V
    discrete_observation_size = [q_table_dimension] * observation_size
    discrete_observation_steps = (env.observation_space.high - env.observation_space.low) / discrete_observation_size

    q_table_size = discrete_observation_size + [env.action_space.n]
    q_table = np.random.uniform(low=-2, high=0, size=(q_table_size)) #(N, N, 3)

    first_goal = True
    show_range = epoch_number // n_show if n_show else epoch_number + 1
    save_range = epoch_number // 100 if epoch_number > 100 else 1
    epsilon_decay_value = epsilon / (epsilon_decaying_range[1] - epsilon_decaying_range[0])

    def get_discrete_state(state): # the indexes about what state we are
        discrete_state = (state - env.observation_space.low) / discrete_observation_steps
        return tuple(discrete_state.astype(np.int))

    def get_continuous_state(state):
        continuous_state = (state + env.observation_space.low) * discrete_observation_steps
        return tuple(discrete_state.astype(np.float))

    start = time.time()
    for epoch in range(1, epoch_number+1):
        #print(epoch)

        epoch_reward = 0

        discrete_state = get_discrete_state(env.reset())


        render = True if not epoch % show_range else False

        done = False
        while not done:
            if np.random.random() > epsilon:
                action = np.argmax(q_table[discrete_state])
            else:
                action = np.random.randint(0, env.action_space.n)

            new_state, reward, done, env_msg = env.step(action)

            epoch_reward += reward

            new_discrete_state = get_discrete_state(new_state)

            if render:
                env.render()

            if not done:
                max_future_q = np.max(q_table[new_discrete_state])
                current_q = q_table[discrete_state + (action,  )]

                new_q = (1 - learning_rate) * current_q + learning_rate * (reward + discount * max_future_q)
                q_table[discrete_state + (action, )] = new_q
            elif new_state[0] >= env.goal_position:
                if first_goal:
                    print("Reached the goal on epoch", epoch)
                    first_goal = False
                q_table[discrete_state + (action, )] = 0

            discrete_state = new_discrete_state

        if epsilon_decaying_range[0] <= epoch <= epsilon_decaying_range[1]:
            epsilon -= epsilon_decay_value

        epoch_rewards.append(epoch_reward)

        if not epoch % save_range:
            average_reward = sum(epoch_rewards[-save_range:])/len(epoch_rewards[-save_range:])
            aggr_ep_rewards["epoch"].append(epoch)
            aggr_ep_rewards["avg_value"].append(average_reward)
            aggr_ep_rewards["min_value"].append(min(epoch_rewards[-save_range:]))
            aggr_ep_rewards["max_value"].append(max(epoch_rewards[-save_range:]))

            #print("Epoch:", epoch, "average:", average_reward, "min:", min(epoch_rewards[-save_range:]), "max:", max(epoch_rewards[-save_range:]))
    print("tot time:", time.time()-start)

    env.close()
    plot_graph_result(test_id, aggr_ep_rewards["epoch"], aggr_ep_rewards["avg_value"], aggr_ep_rewards["max_value"], aggr_ep_rewards["min_value"])
    plot_heat_map(test_id, q_table)


def main():
    jn.create_dir("imgs")
    for test in TEST_LIST:
        launch_new_q_learning_test(
            test["id"],
            test["epochs"],
            test["show_n_time"],
            test["q_table_dimension"],
            test["learning_rate"],
            test["discount"],
            test["epsilon"],
            test["epsilon_decaying_range"]
        )

if __name__ == '__main__':
    main()
