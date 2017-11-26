# README #

===========================================================================================================

Data:
state_responses.csv contains data to be training.
state_responses_test.csv contains data for testing.

===========================================================================================================

CNN Implementations:

2048cnn.py is the trainer py file.
2048-gym_simulator.py is the main simulator py file.

Commands to run:
cd path-to-repo/2048-cnn/
python 2048-gym_simulator.py

===========================================================================================================

RandomForest Implementations:

random-forest-train.py is the trainer py file
2048-gym_simulator_with_randomforest.py is the main simulator py file.

Commands to run:
cd path-to-repo/2048-cnn/
python 2048-gym_simulator_with_randomforest.py

===========================================================================================================

Results:

Games Played : 100

RandomForest Implementation:
Avg Moves: 137.41
Avg Points: 1449.08
Scores Frequency: {64: 30, 128: 43, 256: 22, 32: 5}


CNN Implementation:
Avg Moves: 106.21
Avg Points: 1016.84
Scores Frequency: {64: 46, 128: 33, 32: 9, 16: 2, 256: 10}


Results are not that encouraging as clearly this problem statement is not suited for NeuralNets.
Naturally results of random forest is way better than CNNs but it is below par :(
