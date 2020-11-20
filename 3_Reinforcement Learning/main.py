import os
import gym
import numpy as np
#os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2' # lighter system log
import config as cfg
import janitor as jn

# ACTIONS
GO_LEFT, STAY, GO_RIGHT = 0, 1, 2

Q_TABLE_DIMENSION = 20
LEARNING_RATE = 0.1
DISCOUNT = 0.95
EPSILON = 0.5
EPISODES = 30000
SHOW_EVERY = EPISODES // 50
START_EPSILON_DECAYING = 1
END_EPSILON_DECAYING = EPISODES // 2
EPSILON_DECAY_VALUE = EPSILON / (END_EPSILON_DECAYING - START_EPSILON_DECAYING)

def main():
    np.random.seed(1)
    env = gym.make("MountainCar-v0")
    epsilon = EPSILON

    observation_size = len(env.observation_space.high)
    discrete_observation_size = [Q_TABLE_DIMENSION] * observation_size
    discrete_observation_steps = (env.observation_space.high - env.observation_space.low) / discrete_observation_size

    q_table_size = discrete_observation_size + [env.action_space.n]
    q_table = np.random.uniform(low=-2, high=0, size=(q_table_size)) #(20, 20, 3)

    def get_discrete_state(state): # the indexes about what state we are
        discrete_state = (state - env.observation_space.low) / discrete_observation_steps
        return tuple(discrete_state.astype(np.int))

    for episode in range(EPISODES):
        discrete_state = get_discrete_state(env.reset())

        if episode % SHOW_EVERY == 0:
            print(episode)
            render = True
        else:
            render = False

        done = False
        while not done:
            if np.random.random() > epsilon:
                action = np.argmax(q_table[discrete_state])
            else:
                action = np.random.randint(0, env.action_space.n)





            new_state, reward, done, env_msg = env.step(action)
            new_discrete_state = get_discrete_state(new_state)

            if render:
                env.render()

            if not done:
                max_future_q = np.max(q_table[new_discrete_state])
                current_q = q_table[discrete_state + (action,  )]

                new_q = (1 - LEARNING_RATE) * current_q + LEARNING_RATE * (reward + DISCOUNT * max_future_q)
                q_table[discrete_state + (action, )] = new_q
            elif new_state[0] >= env.goal_position:
                print("Reached the goal on episode", episode)
                q_table[discrete_state + (action, )] = 0

            discrete_state = new_discrete_state

        if START_EPSILON_DECAYING <= episode <= END_EPSILON_DECAYING:
            epsilon -= EPSILON_DECAY_VALUE

    env.close()


 
if __name__ == '__main__':
    main()