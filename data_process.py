import numpy as np
from DataHandler import DataHandler

data_handler = DataHandler(16, 4, "state_responses.csv")

status = True
count = 1
while status:
    status, features, labels = data_handler.get_next_batch(20000)
    print status
    print np.alen(features)
    print np.alen(labels)
    print str(count)
    print "\n\n"
    count = count + 1