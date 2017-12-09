import numpy as np
import scipy.io as spio

def load_Yale_data():
    data = spio.loadmat("data/ExtendedYaleB.mat")
    labels = data['EYALEB_LABEL']
    pictures = data['EYALEB_DATA']
    return pictures, data

if __name__=="__main__":
    pass
