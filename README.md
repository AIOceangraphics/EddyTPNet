# EddyTPNet
This code is from the paper ''Knowledge-fused deep neural network for trajectory prediction of stabilized oceanic mesoscale eddy''
# Dataset
The materials used in this study are obtained from the Satellite Oceano-graphic Data (AVISO+) product(https://data.aviso.altimetry.fr/aviso-gateway/data/META3.1exp_DT/), which cover the years from 1993 to 2020 and include amplitude, radius, speed-average, latitude, longitude, and time data. However, the size of the pre-processed data is too large so that we could notupload the data. Therefore, please download the data for yourself.
# Code Structure
The whole code included eight parts. "Data_loader.py" is used to load data, "DDGEP.py" is the process of processing DDGEP, "GD.py" includes the calculation of earth distance, "GD_Loss.py" is one of the loss functions used in the model, "Moudle.py" describes the network structure, "process.py" includes the training and testing process, "Stand_Normal.py" calculates data normalization, and the "main.py" file is the entry.
