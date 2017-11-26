import gym
from gym import wrappers
import gym_2048
import numpy as np
from DataHandler import DataHandler
import random
import pickle
from sklearn.ensemble import RandomForestClassifier

PREV_STATE_STORE_SIZE = 10
data_handler = DataHandler(16, 4, "")
env = gym.make("2048-v0")
move_mapper_dict = {0: 0, 1: 2, 2: 3, 3: 1}
move_name = ["UP", "RIGHT", "DOWN", "LEFT"]
scores = []

# load the model from disk
saved_model_name = "2048_randomforest_model.sav"
model = pickle.load(open(saved_model_name, 'rb'))

num_of_games = 100
current_game = 1
while current_game <= num_of_games:
    has_game_ended = False
    observation = env.reset()
    prev_state_responses = []
    while not has_game_ended:
        print(env.env.get_board())
        state = [data_handler.treat_features(env.env.get_board().flatten())]
        state_string = str(state)
        prediction = model.predict(state)[0]
        previously_encountered = True
        while previously_encountered:
            state_response = state_string + "#" + str(prediction)
            previously_encountered = state_response in prev_state_responses
            if not previously_encountered:
                prev_state_responses.append(state_response)
            else:
                prediction = (prediction + 1) % 4

        prediction = move_mapper_dict.get(prediction)
        action = prediction
        print("Move Played: " + move_name[action] + "\n\n")

        if len(prev_state_responses) > PREV_STATE_STORE_SIZE:
            prev_state_responses.pop(0)

        action = prediction  # your agent here (this takes random actions)
        observation, reward, done, num_of_moves = env.step(action=action)
        if done:
            env.render()
            env.reset()
            has_game_ended = True

            # Record highest scores
            scores.append((reward, num_of_moves, np.max(observation)))
            break

    print("#" + str(current_game) + " done...")
    current_game = current_game + 1


env.close()

total_moves = 0
total_points = 0
scores_frequency = {}
for score in scores:
    total_points = total_points + score[0]
    total_moves = total_moves + score[1]
    highest = score[2]
    score_frequency = 0
    if highest in scores_frequency:
        score_frequency = scores_frequency[highest]

    score_frequency = score_frequency + 1
    scores_frequency[highest] = score_frequency

print "Avg Moves: " + str(total_moves * 1.0 / len(scores))
print "Avg Points: " + str(total_points * 1.0 / len(scores))
print "Scores Frequency: " + str(scores_frequency)
