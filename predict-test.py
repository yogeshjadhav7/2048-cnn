import numpy as np
from DataHandler import DataHandler
import Predictor as predictor

data_handler = DataHandler(16, 4, "state_responses_vsmall.csv")
features, labels = data_handler.extract_features_labels()
print np.shape(features)
predictions = predictor.predict(features).pop()

print predictions