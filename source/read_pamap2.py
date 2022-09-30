import os
import pandas as pd
import numpy as np
from source.utils import splitOverlapWindows, removeNanWindows

list_of_files = ['PAMAP2_Dataset/Protocol/subject101.dat',
                 'PAMAP2_Dataset/Protocol/subject102.dat',
                 'PAMAP2_Dataset/Protocol/subject103.dat',
                 'PAMAP2_Dataset/Protocol/subject104.dat',
                 'PAMAP2_Dataset/Protocol/subject105.dat',
                 'PAMAP2_Dataset/Protocol/subject106.dat',
                 'PAMAP2_Dataset/Protocol/subject107.dat',
                 'PAMAP2_Dataset/Protocol/subject108.dat',
                 'PAMAP2_Dataset/Protocol/subject109.dat' ]


subjectID = [1,2,3,4,5,6,7,8,9]

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

colNames = ["timestamp", "activityID","heartrate"]

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
    # dataCollection = dataCollection.interpolate() #removal of any remaining NaN value cells by constructing new data points in known set of data points

    return dataCollection

# Data was taken with 100 Hz sampling rate
def read_pamap2(seconds = 4, overlap=0.75, scolumns = ['handAcc16_1', 'handAcc16_2', 'handAcc16_3', 'handAcc6_1', 'handAcc6_2', 'handAcc6_3', 'handGyro1', 'handGyro2', 'handGyro3', 'handMagne1', 'handMagne2', 'handMagne3',]):
    wsize = seconds * 100
    # Reading all  the files and removing orientations
    usersData = {}
    for file in list_of_files:
        procData = pd.read_table(os.path.join('datasets',file), header=None, sep='\s+')
        procData.columns = columns
        procData = dataCleaning(procData)
        usersData[file[-5]] = procData
        
    # Getting the windows
    windowsMap = {}
    X = []
    y = []
    I = []
    allWindows = []
    for userId in subjectID:
        windowsMap[userId] = {}
        df = usersData[str(userId)]
        for actId in activityIDdict.keys():
            dfAct = df.loc[df['activityID'] == actId]
            dfAct = dfAct[scolumns]
            actData = dfAct.values
            windows = splitOverlapWindows(actData, wsize, overlap=overlap)
            # print(windows.shape)
            windowsMap[userId][actId] = removeNanWindows(windows)
            # windowsMap[userId][actId] = windows
            # print(windowsMap[userId][actId].shape)
            win_y = np.array([actId] * len(windowsMap[userId][actId]))
            win_I = np.array([userId] * len(windowsMap[userId][actId]))
            
            if len(windowsMap[userId][actId]) == 0:
                continue
            
            if len(X) == 0:
                X = windowsMap[userId][actId]
                y = win_y
                I = win_I
            else:
                X = np.concatenate((X, windowsMap[userId][actId]), axis = 0)
                y = np.concatenate((y, win_y), axis = 0)
                I = np.concatenate((I, win_I), axis = 0)
    return X, y, I