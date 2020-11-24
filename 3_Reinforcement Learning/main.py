import gym
import numpy as np
import matplotlib.pyplot as plt
import time
import config as cfg
import janitor as jn


# ACTIONS
GO_LEFT, STAY, GO_RIGHT = 0, 1, 2


def plot_graph_result(test_name, epoch_list, avg_list, max_list, min_list, show=False):
    plt.plot(epoch_list, avg_list, label="avg", color="green")
    plt.plot(epoch_list, max_list, label="max", color="red")
    plt.plot(epoch_list, min_list, label="min", color="blue")
    plt.legend(loc='upper left')

    date = time.strftime("%Y_%m_%d_%H_%M_%S")
    plt.savefig(f"imgs\\graph_test_{test_name}_{date}")

    if show:
        plt.show()

    plt.clf()


def plot_heat_map(test_name, q_table, show=False):
    heatmap = np.max(q_table, axis=2)
    plt.imshow(heatmap, cmap='jet', interpolation='nearest', extent=[-0.07, 0.07, 0.6, -1.2], aspect='auto')
    plt.title("State Value function")
    plt.xlabel("Speed (-0.07 to 0.07)")
    plt.ylabel("Position (-1.2 to 0.6)")
    plt.gca().invert_yaxis()
    plt.colorbar()

    date = time.strftime("%Y_%m_%d_%H_%M_%S")
    plt.savefig(f"imgs\\heatmap_test_{test_name}_{date}")

    if show:
        plt.show()

    plt.clf()


def launch_new_q_learning_test(test_id, epoch_number, n_show, q_table_dimension, learning_rate, discount, epsilon, epsilon_decaying_range):
    print(f"\nNew test {test_id}")

    epoch_rewards = []
    test_result = {
        "epoch": [],
        "avg_value": [],
        "min_value": [],
        "max_value": []
    }

    np.random.seed(1)
    env = gym.make("MountainCar-v0")
    observation_size = len(env.observation_space.high)  # S, V
    discrete_observation_size = [q_table_dimension] * observation_size
    q_table_size = discrete_observation_size + [env.action_space.n]  # + A
    q_table = np.random.uniform(low=-2, high=0, size=(q_table_size)) #(N, N, 3)

    epsilon_decay_value = epsilon / (epsilon_decaying_range[1] - epsilon_decaying_range[0])

    discrete_observation_steps = (env.observation_space.high - env.observation_space.low) / discrete_observation_size

    def transform_discrete_state(state): # transform continuos state in the indexes about in which state we are
        discrete_state = (state - env.observation_space.low) / discrete_observation_steps
        return tuple(discrete_state.astype(np.int))

    first_goal = True
    show_range = epoch_number // n_show if n_show else epoch_number + 1
    save_range = epoch_number // 100 if epoch_number > 100 else 1
    goal_counter = 0
    start = time.time()
    for epoch in range(1, epoch_number+1):

        epoch_reward = 0

        current_discrete_state = transform_discrete_state(env.reset())

        render = True if not epoch % show_range else False

        done = False
        while not done:
            # there's a probability of epsilon that there is only random exploration
            action = np.argmax(q_table[current_discrete_state]) if np.random.random() > epsilon else np.random.randint(0, env.action_space.n)

            new_state, reward, done, env_msg = env.step(action)
            epoch_reward += reward

            new_discrete_state = transform_discrete_state(new_state)
            index_q = current_discrete_state + (action,)  # (Pi, Vi, Ai)

            if render:
                env.render()

            if not done:  # udate q_table with the Q function
                max_Q = np.max(q_table[new_discrete_state])
                current_Q = q_table[index_q]
                new_Q = (1 - learning_rate) * current_Q + learning_rate * (reward + discount * max_Q)
                q_table[index_q] = new_Q

            elif new_state[0] >= env.goal_position:  # if the car reached the flag
                goal_counter += 1
                if first_goal:
                    print("Reached the goal on epoch", epoch)
                    first_goal = False
                q_table[index_q] = 0  # best value

            current_discrete_state = new_discrete_state

        if epsilon_decaying_range[0] <= epoch <= epsilon_decaying_range[1]:
            epsilon -= epsilon_decay_value

        epoch_rewards.append(epoch_reward)

        if not epoch % save_range:
            average_reward = sum(epoch_rewards[-save_range:])/len(epoch_rewards[-save_range:])
            test_result["epoch"].append(epoch)
            test_result["avg_value"].append(average_reward)
            test_result["min_value"].append(min(epoch_rewards[-save_range:]))
            test_result["max_value"].append(max(epoch_rewards[-save_range:]))

    print("tot time:", time.time()-start)
    print("tot goal reached:", goal_counter)

    env.close()
    plot_graph_result(test_id, test_result["epoch"], test_result["avg_value"], test_result["max_value"], test_result["min_value"])
    plot_heat_map(test_id, q_table)


def main():
    jn.create_dir("imgs")
    for test in cfg.TEST_LIST:
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
