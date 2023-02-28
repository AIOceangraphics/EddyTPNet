from DDGEP import *
from process import *

if __name__ == '__main__':
    save_Grid('META3.1exp_DT_allsat_Anticyclonic_long_19930101_20200307.nc', 'DDGEP.npy')
    train_test('long_stable_data.npy', 'DDGEP.npy')
