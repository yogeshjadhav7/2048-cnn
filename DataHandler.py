import numpy as np
import pandas as pd


class DataHandler(object):

    def __init__(self, number_of_features, number_of_labels, input_file_path):
        self.NUMBER_OF_FEATURES = number_of_features
        self.NUMBER_OF_LABELS = number_of_labels
        self.POLL_INDEX = 0
        self.FILE_PATH = input_file_path
        print("Data handler object created...")

    def extract_features_labels(self, skip_top_rows=1, shuffle=True):
        data_file = pd.read_csv(self.FILE_PATH, skiprows=np.arange(skip_top_rows), header=None, skip_blank_lines=True)
        data = np.int32(data_file.values)

        if shuffle:
            np.random.shuffle(data)

        features, labels_data = np.hsplit(data, [self.NUMBER_OF_FEATURES])
        labels,_ = np.hsplit(labels_data, [self.NUMBER_OF_LABELS])
        return self.treat_features(features), self.treat_labels(labels)

    def get_next_batch(self, batch_size=100, skip_top_rows=1):
        poll_index = self.POLL_INDEX + batch_size
        data_file = pd.read_csv(self.FILE_PATH, skiprows=np.arange(skip_top_rows), header=None, skip_blank_lines=True)
        data = np.int32(data_file.values)
        len = np.alen(data)
        if self.POLL_INDEX >= len:
            return False, 0, 0

        if poll_index >= len:
            batch_size = len - self.POLL_INDEX

        arr = np.add([i for i in range(batch_size)], self.POLL_INDEX)
        data = np.int32(data[arr, :])
        np.random.shuffle(data)
        data = data[np.random.randint(data.shape[0], size=batch_size), :]
        features, labels_data = np.hsplit(data, [self.NUMBER_OF_FEATURES])
        labels,_ = np.hsplit(labels_data, [self.NUMBER_OF_LABELS])
        self.POLL_INDEX = poll_index
        return True, self.treat_features(features), self.treat_labels(labels)

    def get_random_batch(self, batch_size=100, skip_top_rows=1):
        data_file = pd.read_csv(self.FILE_PATH, skiprows=np.arange(skip_top_rows), header=None, skip_blank_lines=True)
        data = np.int32(data_file.values)
        data = data[np.random.randint(data.shape[0], size=batch_size), :]
        features, labels_data = np.hsplit(data, [self.NUMBER_OF_FEATURES])
        labels,_ = np.hsplit(labels_data, [self.NUMBER_OF_LABELS])
        return self.treat_features(features), self.treat_labels(labels)

    def reset_batch(self):
        self.POLL_INDEX = 0

    def extract_features_labels_generic(self, number_of_features, number_of_labels, input_file_path, skip_top_rows=1, shuffle=True):
        data_file = pd.read_csv(input_file_path, skiprows=np.arange(skip_top_rows), header=None, skip_blank_lines=True)
        data = np.int32(data_file.values)

        if shuffle:
            np.random.shuffle(data)

        features, labels_data = np.hsplit(data, [number_of_features])
        labels,_ = np.hsplit(labels_data, [number_of_labels])
        return self.treat_features(features), self.treat_labels(labels)

    def treat_features(self, features):
        return np.int32(np.log2(np.add(features, 1)))

    def treat_labels(self, labels):
        return np.int32(labels)