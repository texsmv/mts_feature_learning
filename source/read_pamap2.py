import os
import pandas as pd
import numpy as np
from source.utils import splitOverlapWindows, removeNanWindows
from .utils import create_dir

activity_intensity_map = {
    0: "Sedentary",
    1: "Light",
    2: "Moderate",
    3: "Vigorous"
}

list_of_files = ['PAMAP2_Dataset/Protocol/subject101.dat',
                 'PAMAP2_Dataset/Protocol/subject102.dat',
                 'PAMAP2_Dataset/Protocol/subject103.dat',
                 'PAMAP2_Dataset/Protocol/subject104.dat',
                 'PAMAP2_Dataset/Protocol/subject105.dat',
                 'PAMAP2_Dataset/Protocol/subject106.dat',
                 'PAMAP2_Dataset/Protocol/subject107.dat',
                 'PAMAP2_Dataset/Protocol/subject108.dat',
                 'PAMAP2_Dataset/Protocol/subject109.dat' ]

optional_files= ['PAMAP2_Dataset/Optional/subject101.dat',
                 'PAMAP2_Dataset/Optional/subject105.dat',
                 'PAMAP2_Dataset/Optional/subject106.dat',
                 'PAMAP2_Dataset/Optional/subject108.dat',
                 'PAMAP2_Dataset/Optional/subject109.dat' ]


subjectID = [1,2,3,4,5,6,7,8,9]
# Activities compemdium
# https://cdn-links.lww.com/permalink/mss/a/mss_43_8_2011_06_13_ainsworth_202093_sdc1.pdf
activities_met_map = {
    1: 1.0, # Lying quietly
    2: 1.5, # Sitting writting, desk work, tipping
    3: 1.5, # Standing talkingor talking on the phone
    4: 3.5, # Walking moderate speed
    5: 7.5, # Running
    6: 4.0, # Cycling
    7: 5.5, # Nordic walking
    9: 1.3, # Watching TV
    10: 1.5, # Computer work
    11: 2, # Car driving
    12: 8.0, # Ascending stairs
    13: 3.5, # Descending stairs
    16: 3.5, # vacuuming general
    17: 1.8, # ironing
    18: 2.0, # folding laundry
    19: 3.3, # house cleaning 2.3 - 3.3
    20: 4.0, # playing soccer
    24: 9.0, # rope jumping (moderate speed)
}

activityIDdict = {0: 'transient',
              1: 'lying',
              2: 'sitting',
              3: 'standing',
              4: 'walking',
              5: 'running',
              6: 'cycling',
              7: 'Nordic_walking',
              9: 'watching_TV',
              10: 'computer_work',
              11: 'car driving',
              12: 'ascending_stairs',
              13: 'descending_stairs',
              16: 'vacuum_cleaning',
              17: 'ironing',
              18: 'folding_laundry',
              19: 'house_cleaning',
              20: 'playing_soccer',
              24: 'rope_jumping' }

colNames = ["timestamp", "activityID", "heartrate"]

IMUhand = ['handTemperature', 
           'handAcc16_1', 'handAcc16_2', 'handAcc16_3', 
           'handAcc6_1', 'handAcc6_2', 'handAcc6_3', 
           'handGyro1', 'handGyro2', 'handGyro3', 
           'handMagne1', 'handMagne2', 'handMagne3',
           'handOrientation1', 'handOrientation2', 'handOrientation3', 'handOrientation4']

IMUchest = ['chestTemperature', 
           'chestAcc16_1', 'chestAcc16_2', 'chestAcc16_3', 
           'chestAcc6_1', 'chestAcc6_2', 'chestAcc6_3', 
           'chestGyro1', 'chestGyro2', 'chestGyro3', 
           'chestMagne1', 'chestMagne2', 'chestMagne3',
           'chestOrientation1', 'chestOrientation2', 'chestOrientation3', 'chestOrientation4']

IMUankle = ['ankleTemperature', 
           'ankleAcc16_1', 'ankleAcc16_2', 'ankleAcc16_3', 
           'ankleAcc6_1', 'ankleAcc6_2', 'ankleAcc6_3', 
           'ankleGyro1', 'ankleGyro2', 'ankleGyro3', 
           'ankleMagne1', 'ankleMagne2', 'ankleMagne3',
           'ankleOrientation1', 'ankleOrientation2', 'ankleOrientation3', 'ankleOrientation4']

columns = colNames + IMUhand + IMUchest + IMUankle 

def dataCleaning(dataCollection):
    dataCollection = dataCollection.drop(['handOrientation1', 'handOrientation2', 'handOrientation3', 'handOrientation4',
                                            'chestOrientation1', 'chestOrientation2', 'chestOrientation3', 'chestOrientation4',
                                            'ankleOrientation1', 'ankleOrientation2', 'ankleOrientation3', 'ankleOrientation4'],
                                            axis = 1)  # removal of orientation columns as they are not needed
    dataCollection = dataCollection.drop(dataCollection[dataCollection.activityID == 0].index) #removal of any row of activity 0 as it is transient activity which it is not used
    # dataCollection = dataCollection.apply(pd.to_numeric, errors = 'coerse') #removal of non numeric data in cells
    dataCollection = dataCollection.interpolate() #removal of any remaining NaN value cells by constructing new data points in known set of data points

    return dataCollection

PAMAP2_SIGNALS = ['handAcc16_1', 'handAcc16_2', 'handAcc16_3', 'handAcc6_1', 'handAcc6_2', 'handAcc6_3', 'handGyro1', 'handGyro2', 'handGyro3', 'handMagne1', 'handMagne2', 'handMagne3', 'heartrate']

# Data was taken with 100 Hz sampling rate
def read_pamap2(seconds = 4, overlap=0.75, scolumns = PAMAP2_SIGNALS, train_ids = None, test_ids = None, cache=True):
    wsize = seconds * 100
    if train_ids is None:
        ids = subjectID
        shuffled_indices = np.random.permutation(len(ids))
        ids = [ids[i] for i in shuffled_indices]
        
        test_no = int(0.4 * len(ids))
        n = len(ids)
        
        train_ids = ids[:n - test_no]
        test_ids = ids[n - test_no:]
    # Reading all  the files and removing orientations
    
    dataset_name = 'PAMAP_test_{}_wsize_{}_overlap_{}_scolumns{}.npy'.format('_'.join([str(i) for i in test_ids]), wsize, overlap, '_'.join([i for i in scolumns]))
    
    DB_CACHE_PATH = os.path.join('cache/', dataset_name)
    create_dir('cache')
    
    if cache and os.path.exists(DB_CACHE_PATH):
        # print('Loading dataset from cache...')
        return np.load(DB_CACHE_PATH, allow_pickle=True)[()]
    
    usersData = {}
    for file in list_of_files:
        procData = pd.read_table(os.path.join('datasets',file), header=None, sep='\s+')
        procData.columns = columns
        procData = dataCleaning(procData)
        usersData[file[-5]] = procData
        
    # Getting the windows
    windowsMap_train = {}
    X_train = []
    y_train = []
    I_train = []
    for userId in train_ids:
        windowsMap_train[userId] = {}
        df = usersData[str(userId)]
        for actId in activityIDdict.keys():
            dfAct = df.loc[df['activityID'] == actId]
            dfAct = dfAct[scolumns]
            actData = dfAct.values
            windows = splitOverlapWindows(actData, wsize, overlap=overlap)
            # print(windows.shape)
            windowsMap_train[userId][actId] = removeNanWindows(windows)
            # windowsMap_train[userId][actId] = windows
            # print(windowsMap_train[userId][actId].shape)
            win_y = np.array([actId] * len(windowsMap_train[userId][actId]))
            win_I = np.array([userId] * len(windowsMap_train[userId][actId]))
            
            if len(windowsMap_train[userId][actId]) == 0:
                continue
            
            if len(X_train) == 0:
                X_train = windowsMap_train[userId][actId]
                y_train = win_y
                I_train = win_I
            else:
                X_train = np.concatenate((X_train, windowsMap_train[userId][actId]), axis = 0)
                y_train = np.concatenate((y_train, win_y), axis = 0)
                I_train = np.concatenate((I_train, win_I), axis = 0)
    
    windowsMap_test = {}
    X_test = []
    y_test = []
    I_test = []
    for userId in test_ids:
        windowsMap_test[userId] = {}
        df = usersData[str(userId)]
        for actId in activityIDdict.keys():
            dfAct = df.loc[df['activityID'] == actId]
            dfAct = dfAct[scolumns]
            actData = dfAct.values
            windows = splitOverlapWindows(actData, wsize, overlap=overlap)
            # print(windows.shape)
            windowsMap_test[userId][actId] = removeNanWindows(windows)
            # windowsMap_train[userId][actId] = windows
            # print(windowsMap_train[userId][actId].shape)
            win_y = np.array([actId] * len(windowsMap_test[userId][actId]))
            win_I = np.array([userId] * len(windowsMap_test[userId][actId]))
            
            if len(windowsMap_test[userId][actId]) == 0:
                continue
            
            if len(X_test) == 0:
                X_test = windowsMap_test[userId][actId]
                y_test = win_y
                I_test = win_I
            else:
                X_test = np.concatenate((X_test, windowsMap_test[userId][actId]), axis = 0)
                y_test = np.concatenate((y_test, win_y), axis = 0)
                I_test = np.concatenate((I_test, win_I), axis = 0)
                
    X_train = X_train.astype(np.float32)
    y_train = y_train.astype(np.int32)
    I_train = I_train.astype(np.int32)
    
    X_test = X_test.astype(np.float32)
    y_test = y_test.astype(np.int32)
    I_test = I_test.astype(np.int32)
    
    data_map = {
        'train': [train_ids, X_train, y_train, I_train],
        'test': [test_ids, X_test, y_test, I_test],
    }
    np.save(DB_CACHE_PATH, data_map)
    return data_map



# Modes: leave-one-subject, shuffle-all
class DatasetPAMAP2:
    def __init__(self, mode, seconds = 4, overlap = 0.5):
        self.mode = mode
        self.activities = list(activityIDdict.values())
        self.activities_map = activityIDdict
        self.users = subjectID
        self.intensity_map = activity_intensity_map
        self.signals = np.array(PAMAP2_SIGNALS)
        self.currFold = -1
        self.N_TESTS = 1
        self.seconds = seconds
        self.overlap = overlap 
        self.intensities = [
            "Sedentary",
            "Light",
            "Moderate",
            "Vigorous"
        ]

    def filterSignals(self, signals):
        signals = np.array(signals)
        
        idx = [np.where(self.signals == dim)[0][0] for dim in signals]
        self.X_train = self.X_train[:, :, idx]
        self.X_test = self.X_test[:, :, idx]
        

    def loadData(self):
        if self.mode == 'leave-one-subject':
            self.currFold = self.currFold + 1
            if self.currFold == len(self.users):
                return False
            self.test_users = self.users[self.currFold: self.currFold + self.N_TESTS]
            self.train_users = self.users[:self.currFold] + self.users[self.currFold + self.N_TESTS:]
        else:
            self.train_users = self.users
            self.test_users = []
        
        data = read_pamap2(
            train_ids=self.train_users, 
            test_ids=self.test_users, 
            cache=True,
            overlap=self.overlap,
            seconds=self.seconds,
        )
        
        if self.mode == 'leave-one-subject':
            _, self.X_train, self.y_train, self.I_train = data['train']
            self.MET_train = np.array([ activities_met_map[act] for act in self.y_train])
            self.Int_train = np.zeros(self.X_train.shape[0])
            self.Int_train[self.MET_train > 6.0] = 3
            self.Int_train[self.MET_train <= 6.0] = 2
            self.Int_train[self.MET_train <= 3.0] = 1
            self.Int_train[self.MET_train <= 1.5] = 0
            
            _, self.X_test, self.y_test, self.I_test = data['test']
            self.MET_test = np.array([ activities_met_map[act] for act in self.y_test])
            self.Int_test = np.zeros(self.X_test.shape[0])
            self.Int_test[self.MET_test > 6.0] = 3
            self.Int_test[self.MET_test <= 6.0] = 2
            self.Int_test[self.MET_test <= 3.0] = 1
            self.Int_test[self.MET_test <= 1.5] = 0
        return True


