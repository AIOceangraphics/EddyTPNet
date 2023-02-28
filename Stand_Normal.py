import numpy as np

def Standardize(data):
    a, b, c = data.shape
    data = data.reshape(a * b, c)
    data_mean = np.mean(data, axis=0)
    data_std = np.std(data, axis=0)
    data_stand = np.divide(np.subtract(data, data_mean), data_std)
    data_stand = data_stand.reshape(a, b, c)
    return data_stand,data_mean,data_std

def abStandardize(data_stand,data_mean,data_std):
    data = data_stand * data_std + data_mean
    return data

def Normalization(data):
    a, b, c = data.shape
    data = data.reshape(a * b, c)
    data_max = np.max(data, axis=0)
    data_min = np.min(data, axis=0)
    data_normal = (data - data_min) / (data_max - data_min)
    data_normal = data_normal.reshape(a, b, c)
    return data_normal, data_max, data_min

def abNormalization(data_normal, data_max, data_min):
    data = data_normal * (data_max - data_min) + data_min
    return data