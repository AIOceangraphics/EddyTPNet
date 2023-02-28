import torch
from torch.utils.data import DataLoader, TensorDataset

def Data_loader(train_data,train_label,test_data,test_label,batch_size):
    dataset_train = TensorDataset(torch.tensor(train_data.astype('float64'), dtype=torch.float),
                                  torch.tensor(train_label.astype('float64'), dtype=torch.float))
    loader_train = DataLoader(dataset_train, shuffle=True, batch_size=batch_size)

    dataset_test = TensorDataset(torch.tensor(test_data.astype('float64'), dtype=torch.float),
                                 torch.tensor(test_label.astype('float64'), dtype=torch.float))
    loader_test = DataLoader(dataset_test, shuffle=False, batch_size=batch_size)

    return loader_train,loader_test