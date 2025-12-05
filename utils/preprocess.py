import numpy as np

def make_window(series, window_size):
    X = []
    for i in range(len(series) - window_size):
        X.append(series[i:i+window_size])
    return np.array(X)