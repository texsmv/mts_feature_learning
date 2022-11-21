# This code was modified from the original repository: HAR-UML20
import csv
import os
import numpy as np
from sklearn.utils import shuffle
from .utils import create_dir

har_dimensions = np.array([
    'Accelerometer-X',	
    'Accelerometer-Y',	
    'Accelerometer-Z',
    'Gyrometer-X',
    'Gyrometer-Y',
    'Gyrometer-Z',
    'Magnetometer-X',
    'Magnetometer-Y',
    'Magnetometer-Z'
])

har_activities_map = {
    0: "Sitting",
    1: "Lying",
    2: "Standing",
    3: "Walking",
    4: "Running",
    5: "Downstairs",
    6: "Upstairs"
}

activity_intensity_map = {
    0: "Sedentary",
    1: "Light",
    2: "Moderate",
    3: "Vigorous"
}

har_activities = np.array([
    "Sitting",
    "Lying",
    "Standing",
    "Walking",
    "Running",
    "Downstairs",
    "Upstairs"
])

def csv_parse(filename, path_to_dir):
    # print("\nReading file: {}".format(filename))
    fullpath = path_to_dir + '/' + filename # str format
    headerline = 2
    raw_list = []
    headerlist = []

    with open(fullpath) as csvfile:
        readCSV = csv.reader(csvfile, delimiter=',')
        i=0

        for row in readCSV:
            if i < headerline: # headerline
                headerlist.extend([row])
                i += 1
            else:
                raw_list.extend([row])
        raw_array = np.asarray(raw_list)
    return headerlist, raw_array


def find_csv_filenames( path_to_dir, suffix=".csv" ):
    filenames = os.listdir(path_to_dir)
    return [ filename for filename in filenames if filename.endswith( suffix ) ]


def get_dataset(path_to_dir):
	data = find_csv_filenames(path_to_dir)
	data = [ file for file in data if 'dataset' in file]
	epochslist = find_csv_filenames(path_to_dir, 'epochs.csv')
	no_data = len(epochslist)
	d_list = []
	e_list = []
	for i in range(1, int(no_data+1)):
		for d in data:
			name = 'S'+str(i)+'.'
			if (name in d) & ('epochs' not in d):
				d_list.append(d)
				e_list.append(d[:-4]+'_epochs.csv')
	return (d_list, e_list)


har_ind_IDS = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9]

def read_har_dataset(path_to_dir, train_ids = None, test_ids = None, val_ids = None, cache=True):
    
    allfilenames, epochnames = get_dataset(path_to_dir)
    ids = list(range(len(allfilenames)))
    if train_ids is None:
        shuffled_indices = np.random.permutation(len(ids))
        ids = [ids[i] for i in shuffled_indices]
        
        test_no = int(0.4 * len(allfilenames))
        n = len(allfilenames)
        
        train_ids = ids[:n - test_no]
        testing_ids = ids[n - test_no:] # both test and validation ids
        
        val_ids = testing_ids[int(test_no/2):]
        test_ids = testing_ids[:int(test_no/2)]
        
    
    dataset_name = 'HAR_UML20_test_{}_val_{}.npy'.format('_'.join([str(i) for i in test_ids]), '_'.join([str(i) for i in val_ids]))
    
    DB_CACHE_PATH = os.path.join('cache/', dataset_name)
    create_dir('cache')
    
    
    if cache and os.path.exists(DB_CACHE_PATH):
        # print('Loading dataset from cache...')
        return np.load(DB_CACHE_PATH, allow_pickle=True)[()]
    

    
    # ------------- Apply random permutation---------------
    # allfilenames = [allfilenames[i] for i in shuffled_indices]
    # epochnames = [epochnames[i] for i in shuffled_indices]
    # -----------------------------------------------------
    
    
    # print('IDS: {}'.format(ids))
    
    
    
    X_train , X_test, X_validation, y_train, y_test, y_val = None, None, None, None, None, None
    I_train, I_test, I_val = None, None, None
    
    

    for id in train_ids:
        file = allfilenames[id]
        [_, data] = csv_parse(file, path_to_dir)
        windows = np.reshape(data[:, 6:15], (-1, 200, 9))
        windows_ids = np.ones((windows.shape[0]))
        windows_ids[:] = id
        if X_train is None:
            X_train = windows
            I_train = windows_ids
        else:
            X_train = np.concatenate((X_train, windows), axis=0)
            I_train = np.concatenate((I_train, windows_ids), axis=0)

    """
    [_, data] = csv_parse(filenames[0])
    X = np.reshape(data[:, 6:15], (-1, 200, 9))
    [_, data] = csv_parse(filenames[1])
    X = np.concatenate((X, np.reshape(data[:,6:15], (-1, 200, 9))), axis=0)
    [_, data] = csv_parse(filenames[2])
    X = np.concatenate((X, np.reshape(data[:,6:15], (-1, 200, 9))), axis=0)
    [_, data] = csv_parse(filenames[3])
    X_train = np.concatenate((X, np.reshape(data[:,6:15], (-1, 200, 9))), axis=0)
    """
    epochs, kcal_MET = None, None
    for id in train_ids:
        epoch = epochnames[id]
        [_, data] = csv_parse(epoch, path_to_dir)
        eps = data[:2,8:11]
        km = data[:2,11:13]
        y_ = data[:2,8]
        
        for row in data[:,:13]:
            if row[8]:
                eps = np.append(eps, [row[8:11]], axis=0)
                km = np.append(km, [row[11:]], axis=0)
                if row[2] == "SITTING":
                    y_ = np.concatenate((y_, [0]), axis=0)
                elif row[2] == "LYING":
                    y_ = np.concatenate((y_, [1]), axis=0)
                elif row[2] == "STANDING":
                    y_ = np.concatenate((y_, [2]), axis=0)
                elif row[2] == "WALKING":
                    y_ = np.concatenate((y_, [3]), axis=0)
                elif row[2] == "RUNNING":
                    y_ = np.concatenate((y_, [4]), axis=0)
                elif row[2] == "DOWNSTAIRS":
                    y_ = np.concatenate((y_, [5]), axis=0)
                elif row[2] == "UPSTAIRS":
                    y_ = np.concatenate((y_, [6]), axis=0)
        if y_train is None:
            y_train = np.reshape(y_[2:], (-1, 1))
        else:
            y_train = np.concatenate((y_train, np.reshape(y_[2:], (-1, 1))), axis=0)
        y_ = None

        if epochs is None:
            epochs = np.reshape(eps[2:,:], (-1, 3))
        else:
            epochs = np.concatenate((epochs, np.reshape(eps[2:,:], (-1, 3))), axis=0)
        if kcal_MET is None:
            kcal_MET = np.reshape(km[2:,:], (-1, 2))
        else:
            kcal_MET = np.concatenate((kcal_MET, np.reshape(km[2:,:], (-1, 2))), axis=0)


    
    test_epochs, test_kcal_MET = None, None
    for id in test_ids:
        file = allfilenames[id]
        [_, data] = csv_parse(file, path_to_dir)
        
        windows = np.reshape(data[:, 6:15], (-1, 200, 9))
        windows_ids = np.ones((windows.shape[0]))
        windows_ids[:] = id
        if X_test is None:
            X_test = windows
            I_test = windows_ids
        else:
            X_test = np.concatenate((X_test, windows), axis=0)
            I_test = np.concatenate((I_test, windows_ids), axis=0)

    for id in test_ids:
        epoch = epochnames[id]
        [_, data] = csv_parse(epoch, path_to_dir)
        test_eps = data[:2,8:11]
        test_km = data[:2,11:13]
        y_ = data[:2,8]
        for row in data[:,:13]:
            if row[8]:
                test_eps = np.append(test_eps, [row[8:11]], axis=0)
                test_km = np.append(test_km, [row[11:]], axis=0)
                if row[2] == "SITTING":
                    y_ = np.concatenate((y_, [0]), axis=0)
                elif row[2] == "LYING":
                    y_ = np.concatenate((y_, [1]), axis=0)
                elif row[2] == "STANDING":
                    y_ = np.concatenate((y_, [2]), axis=0)
                elif row[2] == "WALKING":
                    y_ = np.concatenate((y_, [3]), axis=0)
                elif row[2] == "RUNNING":
                    y_ = np.concatenate((y_, [4]), axis=0)
                elif row[2] == "DOWNSTAIRS":
                    y_ = np.concatenate((y_, [5]), axis=0)
                elif row[2] == "UPSTAIRS":
                    y_ = np.concatenate((y_, [6]), axis=0)

        if y_test is None:
            y_test = np.reshape(y_[2:], (-1, 1))
        else:
            y_test = np.concatenate((y_test, np.reshape(y_[2:], (-1, 1))), axis=0)
        y_ = None

        if test_epochs is None:
            test_epochs = test_eps[2:,:]
        else:
            test_epochs = np.concatenate((test_epochs, test_eps[2:,:]), axis=0)
        if test_kcal_MET is None:
            test_kcal_MET = test_km[2:,:]
        else:
            test_kcal_MET = np.concatenate((test_kcal_MET, test_km[2:,:]), axis=0)
    
   

    
    validation_epochs, validation_kcal_MET = None, None
    for id in val_ids:
        file = allfilenames[id]
        [_, data] = csv_parse(file, path_to_dir)
        
        windows = np.reshape(data[:, 6:15], (-1, 200, 9))
        windows_ids = np.ones((windows.shape[0]))
        windows_ids[:] = id
        if X_validation is None:
            X_validation = windows
            I_val = windows_ids
        else:
            X_validation = np.concatenate((X_validation, windows), axis=0)
            I_val = np.concatenate((I_val, windows_ids), axis=0)

    for id in val_ids:
        epoch = epochnames[id]
        [_, data] = csv_parse(epoch, path_to_dir)
        validation_eps = data[:2,8:11]
        validation_km = data[:2,11:13]
        y_ = data[:2,8]
        for row in data[:,:13]:
            if row[8]:
                validation_eps = np.append(validation_eps, [row[8:11]], axis=0)
                validation_km = np.append(validation_km, [row[11:]], axis=0)
                if row[2] == "SITTING":
                    y_ = np.concatenate((y_, [0]), axis=0)
                elif row[2] == "LYING":
                    y_ = np.concatenate((y_, [1]), axis=0)
                elif row[2] == "STANDING":
                    y_ = np.concatenate((y_, [2]), axis=0)
                elif row[2] == "WALKING":
                    y_ = np.concatenate((y_, [3]), axis=0)
                elif row[2] == "RUNNING":
                    y_ = np.concatenate((y_, [4]), axis=0)
                elif row[2] == "DOWNSTAIRS":
                    y_ = np.concatenate((y_, [5]), axis=0)
                elif row[2] == "UPSTAIRS":
                    y_ = np.concatenate((y_, [6]), axis=0)

        if y_val is None:
            y_val = np.reshape(y_[2:], (-1, 1))
        else:
            y_val = np.concatenate((y_val, np.reshape(y_[2:], (-1, 1))), axis=0)
        y_ = None

        if validation_epochs is None:
            validation_epochs = validation_eps[2:,:]
        else:
            validation_epochs = np.concatenate((validation_epochs, validation_eps[2:,:]), axis=0)
        if validation_kcal_MET is None:
            validation_kcal_MET = validation_km[2:,:]
        else:
            validation_kcal_MET = np.concatenate((validation_kcal_MET, validation_km[2:,:]), axis=0)
    
   



    """
    outputs:
    X_train (7770, 200, 9)
    y_train (7770, 1)
    X_test (630, 200, 9)
    y_test (630, 1)
    epochs (630, 3)
    """

    #shuffle training set
    y_train = np.concatenate((y_train.astype(np.float), epochs.astype(np.float), kcal_MET.astype(np.float)), axis=1)
    X_train, y_train, I_train = shuffle(X_train.astype(np.float), y_train, I_train)
    epochs = y_train[:,1:4]   # extract count values
    kcal_MET = y_train[:, 4:] # extract kcal & MET values
    y_train = y_train[:,0].reshape(-1, 1)   # extract class labels
    
    
    
    
    
    kcal_MET = kcal_MET.astype(np.float32)
    X_train = X_train.astype(np.float32)
    y_train = y_train.astype(np.int32)
    I_train = I_train.astype(np.int32)
    
    test_kcal_MET = test_kcal_MET.astype(np.float32)
    X_test = X_test.astype(np.float32)
    y_test = y_test.astype(np.int32)
    I_test = I_test.astype(np.int32)

    if len(val_ids) != 0:
        val_kcal_MET = validation_kcal_MET.astype(np.float32)
        X_val = X_validation.astype(np.float32)
        y_val = y_val.astype(np.int32)
        I_val = I_val.astype(np.int32)
    else:
        val_kcal_MET = X_val = y_val = I_val = np.array([])
    
    data_map = {
        'train': [train_ids, X_train, y_train, I_train, kcal_MET],
        'test': [test_ids, X_test, y_test, I_test, test_kcal_MET],
        'val':[val_ids, X_val, y_val, I_val, val_kcal_MET]
    }
    
    np.save(DB_CACHE_PATH, data_map)
    return data_map

# Modes: leave-one-subject, shuffle-all
class DatasetHARUML20:
    def __init__(self, mode):
        self.mode = mode
        self.signals = np.array(har_dimensions)
        self.activities = har_activities
        self.activities_map = har_activities_map
        self.users = har_ind_IDS
        self.intensity_map = activity_intensity_map
        self.intensities = [
            "Sedentary",
            "Light",
            "Moderate",
            "Vigorous"
        ]        
        self.currFold = -1
        self.N_TESTS = 1
        
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
            
        data = read_har_dataset(
            './datasets/HAR-UML20/', 
            train_ids=self.train_users, 
            test_ids=self.test_users, 
            val_ids=[], 
            cache=True
        )
        
        if self.mode == 'leave-one-subject':
            _, self.X_train, self.y_train, self.I_train, train_kcal_MET = data['train']
            self.MET_train = train_kcal_MET[:,1]
            self.Int_train = np.zeros(self.X_train.shape[0])
            self.Int_train[self.MET_train > 6.0] = 3
            self.Int_train[self.MET_train <= 6.0] = 2
            self.Int_train[self.MET_train <= 3.0] = 1
            self.Int_train[self.MET_train <= 1.5] = 0
            
            _, self.X_test, self.y_test, self.I_test, test_kcal_MET = data['test']
            self.MET_test = test_kcal_MET[:,1]
            self.Int_test = np.zeros(self.X_test.shape[0])
            self.Int_test[self.MET_test > 6.0] = 3
            self.Int_test[self.MET_test <= 6.0] = 2
            self.Int_test[self.MET_test <= 3.0] = 1
            self.Int_test[self.MET_test <= 1.5] = 0
        return True
