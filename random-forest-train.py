import pandas as pd
import numpy as np
import pickle
from sklearn.ensemble import RandomForestClassifier

def data_extractor(filename, gridsize):
    dataframe = pd.read_csv(filename)
    data_array = dataframe.values

    X = data_array[:, 0:gridsize]
    X[X == 0] = 1
    X = np.log2(X)

    Y = np.argmax(data_array[:, gridsize:len(data_array[0])], axis=1)

    return X, Y

testing_file = "state_responses_test.csv"
training_file = "state_responses.csv"
gridsize = 16

test_X, test_Y = data_extractor(testing_file, gridsize)
train_X, train_Y = data_extractor(training_file, gridsize)

model = RandomForestClassifier(bootstrap=False,
                               class_weight=None,
                               criterion='gini',
                               max_features='log2',
                               max_depth=None,
                               min_samples_split=2,
                               min_samples_leaf=1,
                               max_leaf_nodes=None,
                               min_impurity_decrease=0.0,
                               min_weight_fraction_leaf=0.0,
                               n_estimators=5,
                               n_jobs=-1,
                               oob_score=False,
                               random_state=42,
                               verbose=0,
                               warm_start=False)


print "Training on training data..."
print model.fit(train_X, train_Y)


print "Testing on training data..."
print model.score(train_X, train_Y)


print "Testing on testing data..."
print model.score(test_X, test_Y)

# save the model to disk
filename = '2048_randomforest_model.sav'
pickle.dump(model, open(filename, 'wb'))