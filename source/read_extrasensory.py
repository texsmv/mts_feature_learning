import pandas as pd
import numpy as np
import os

def readMagGyr():
    path = 'full_raws_0_gyr_mag_4seconds_mets.csv2000values.csv'
    path = 'full_raws_0_gyr_mag_4seconds_mets.csv2000values.csv'
    
    df = pd.read_csv(os.path.join('datasets/ExtraSensory/', path), header=None)
    
    
    UserIds = df.values[:, 0]
    y = df.values[:, 2]
    intensity_type = df.values[:, 4]
    data  = df.values[:, 5:] # Picking only the gyrometer and magnetometer values: (B, 6 * T)
    T = data.shape[1] // 6
    
    dimensions_names = ['mag_x', 'mag_y', 'mag_z', 'gyr_x', 'gyr_y', 'gyr_z']
    
    dataset = np.zeros((data.shape[0], 6, T))
    
    for n in range(data.shape[0]):
        for d in range(6):
            for t in range(T):
                # print(t * 6 + d)
                dataset[n, d, t] = data[n, t * 6 + d]
    
    Ids = np.unique(UserIds)
    I = np.copy(UserIds)
    for i in range(len(Ids)):
        I[I == Ids[i]] = i
    
    Intensities = np.unique(intensity_type)
    Int = np.copy(intensity_type)
    for i in range(len(Intensities)):
        Int[Int == Intensities[i]] = i
    
    
    return dataset.astype(np.float32), y, I, Int, data, np.array(dimensions_names), np.array(Intensities)

def readAcc():
    path = 'full_raws_0_watch_acc_labels_3values_4seconds_.csv'
    # path = 'full_raws_3_watch_acc_labels_3values_4seconds_.csv'
    # path = 'full_raws_3_watch_acc_labels_3values_4seconds_.csv2000values.csv'
    # df = pd.read_csv(os.path.join('datasets/ExtraSensory/', path), header=None, nrows=100000)
    df = pd.read_csv(os.path.join('datasets/ExtraSensory/', path), header=None)
    
    # print(df.shape)
    df = df.sample(frac=1).reset_index(drop=True)
    df = df.head(100000)
    # print(df.shape)
    
    y = df.values[:, 2]
    data  = df.values[:, 3:] # Picking only the accelerometer values: (B, 3 * T)
    T = data.shape[1] // 3
    df = None
    
    
    dimensions_names = ['acc_x', 'acc_y', 'acc_z']
    
    dataset = np.zeros((data.shape[0], 3, T))
    
    for n in range(data.shape[0]):
        for d in range(3):
            for t in range(T):
                dataset[n, d, t] = data[n, t * 3 + d]
    
    return dataset.astype(np.float32), y, np.array(dimensions_names)