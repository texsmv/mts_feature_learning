import os
from math import floor
import numpy as np
import pandas as pd
import datetime
from .utils import splitWindows, create_dir

DATASET_PATH = 'datasets/OpenDataset/'

# all_activities=  ['acostado-enreposo-cama;', 'acostado-enreposo-sofa;',
#  'acostado-usandotelefono-cama;', 'acostado-usandotelefono-sofa;',
#  'acostado-viendotelevision-sofa;', 'caminando;', 'parado;',
#  'reclinado-enreposo-cama;', 'reclinado-enreposo-escritorio;',
#  'reclinado-enreposo-sofa;', 'reclinado-usandocomputador-cama;',
#  'reclinado-usandocomputador-escritorio;', 'reclinado-usandotelefono-cama;',
#  'reclinado-usandotelefono-escritorio;', 'reclinado-usandotelefono-sofa;',
#  'reclinado-viendotelevision-sofa;', 'sentado-enreposo-cama;',
#  'sentado-enreposo-escritorio;', 'sentado-enreposo-sofa;',
#  'sentado-usandocomputador-cama;', 'sentado-usandocomputador-escritorio;',
#  'sentado-usandotelefono-cama;', 'sentado-usandotelefono-escritorio;',
#  'sentado-usandotelefono-sofa;', 'sentado-viendotelevision-sofa;']

activities=  ['acostado-enreposo-cama;', 'acostado-enreposo-sofa;',
 'acostado-usandotelefono-cama;', 'acostado-usandotelefono-sofa;',
 'acostado-viendotelevision-sofa;', 'caminando;', 'parado;',
 'reclinado-enreposo-cama;', 'reclinado-enreposo-escritorio;',
 'reclinado-enreposo-sofa;', 'reclinado-usandocomputador-cama;',
 'reclinado-usandocomputador-escritorio;', 'reclinado-usandotelefono-cama;',
 'reclinado-usandotelefono-escritorio;', 'reclinado-usandotelefono-sofa;',
 'reclinado-viendotelevision-sofa;', 'sentado-enreposo-cama;',
 'sentado-enreposo-escritorio;', 'sentado-enreposo-sofa;',
 'sentado-usandocomputador-cama;', 'sentado-usandocomputador-escritorio;',
 'sentado-usandotelefono-cama;', 'sentado-usandotelefono-escritorio;',
 'sentado-usandotelefono-sofa;', 'sentado-viendotelevision-sofa;']

met_map = {
    'acostado-enreposo-cama;': 1.0, 
    'acostado-enreposo-sofa;': 1.0,
    'acostado-usandotelefono-cama;': 1.3,
    'acostado-usandotelefono-sofa;': 1.3,
    'acostado-viendotelevision-sofa;': 1.0,
    'caminando;': 3.0, 
    'parado;': 1.5, # assuming quietly
    'reclinado-enreposo-cama;': 1.0,
    'reclinado-enreposo-escritorio;': 1.0,
    'reclinado-enreposo-sofa;': 1.0,
    'reclinado-usandocomputador-cama;': 1.3,
    'reclinado-usandocomputador-escritorio;': 1.5, 
    'reclinado-usandotelefono-cama;': 1.3,
    'reclinado-usandotelefono-escritorio;': 1.3, 
    'reclinado-usandotelefono-sofa;': 1.3,
    'reclinado-viendotelevision-sofa;': 1.0,
    'sentado-enreposo-cama;': 1.0,
    'sentado-enreposo-escritorio;': 1.0,
    'sentado-enreposo-sofa;': 1.0,
    'sentado-usandocomputador-cama;': 1.5,
    'sentado-usandocomputador-escritorio;': 1.5,
    'sentado-usandotelefono-cama;': 1.3,
    'sentado-usandotelefono-escritorio;': 1.3,
    'sentado-usandotelefono-sofa;': 1.3,
    'sentado-viendotelevision-sofa;': 1.0,
}

activities_met_map = {i: met_map[activities[i]]  for i in range(len(activities))}

activities_map = {
    i: activities[i]
    for i in range(len(activities))
}

activity_intensity_map = {
    0: "Sedentary",
    1: "Light",
    2: "Moderate",
    3: "Vigorous"
}

columns = [
    'accelerometer_x',
    'accelerometer_y',
    'accelerometer_z',
    'gyroscope_x',
    'gyroscope_y',
    'gyroscope_z',
    'heart_rate',
    'skin_temperature',
    'galvanic_skin_response',
    'rr_interval',
    'light',
    'barometer',
    'altimeter',
]
individuals = [e for e in range(100, 130)]
individuals.remove(114)
individuals.remove(115)
# individuals.remove(116)
individuals = np.array(individuals)

# Gets the indivuals data as well as the labels per activity from an individual
def get_activity_values(indv, window_size, data_map):
    data = []
    labels = []
    for act in activities:
        act_data = splitWindows(data_map[indv][act], window_size)
        act_labels = np.array([activities.index(act) for e in range(len(act_data))])
        if len(data) == 0:
            data = act_data
            labels = act_labels
        else:
            data = np.concatenate([data, act_data])
            labels = np.concatenate([labels, act_labels])
    
    return data, labels

# Default sampling rate is 50hz
def read_open_dataset(seconds = 4, train_ids = None, test_ids = None, cache=True):
    resample_rule = '20L' # 50hz
    window_size = floor(seconds * 50)
        
    # Individuals ids and training and test separated
    
    val_ids = []
    if train_ids is None:
        idx = np.random.choice(np.arange(len(individuals)), floor(0.4 * len(individuals)), replace=False)
        test_idx = idx[int(len(idx)/2):]
        val_idx = idx[:int(len(idx)/2)]

        test_ids = individuals[test_idx]
        val_ids = individuals[val_idx]
        
        train_ids = []
        for ind in individuals:
            if (ind not in test_ids) and (ind not in val_ids):
                train_ids = train_ids + [ind]
        
        
    train_ids = np.array(train_ids)
    test_ids = np.array(test_ids)
    val_ids = np.array(val_ids)
    
    dataset_name = 'OpenDataset_test_{}_wsize_{}.npy'.format('_'.join([str(i) for i in test_ids]), window_size)
    
    DB_CACHE_PATH = os.path.join('cache/', dataset_name)
    create_dir('cache')
    
    if cache and os.path.exists(DB_CACHE_PATH):
        # print('Loading dataset from cache...')
        return np.load(DB_CACHE_PATH, allow_pickle=True)[()]
    

    # read the data
    folder = 'MainPhone'
    data_df = {}
    for indv in individuals:
        data_df[indv] = pd.read_csv(f'{DATASET_PATH}/SedentaryBehaviorsDataSet/{folder}/{indv}/{indv}.txt', header = None)
    data_map = {}
    

    for indv in individuals:
        idf = data_df[indv]
        data_map[indv] = {}
        for act in activities:
            adf = idf.loc[(idf[41] == act)]
            if adf.to_numpy().shape[1] != 42:
                print("WRONG SHAPE")
                print(adf.to_numpy().shape)
            index = np.array([datetime.datetime.fromtimestamp(e/1000.0) for e in adf[1]])
            # print(adf.head())
            adf = pd.DataFrame(adf.to_numpy()[:, 2:15], index = index) # HR

            adf.columns = columns
            # adf = adf [filters]
            adf = adf.infer_objects()
            adf = adf.resample(resample_rule).mean().ffill().bfill() # resample to 40 Hz
            has_nan = np.isnan(adf.to_numpy()).any()
            if has_nan:
                print("NANs!!!!!")

            data_map[indv][act] = adf.to_numpy()

    X_train = []
    y_train = []
    I_train = []
    for indv in train_ids:
        idata, ilabels = get_activity_values(indv, window_size, data_map)
        n_windows = len(idata) # number of windows for this ind
        I_window = (np.ones(n_windows) * indv).astype(int)
        if len(X_train) == 0:
            X_train = idata
            y_train = ilabels
            I_train  = I_window
        else:
            X_train = np.concatenate([X_train, idata])
            y_train = np.concatenate([y_train, ilabels])
            I_train = np.concatenate([I_train, I_window])
    
    X_test = []
    y_test = []
    I_test = []
    for indv in test_ids:
        idata, ilabels = get_activity_values(indv, window_size, data_map)
        n_windows = len(idata) # number of windows for this ind
        I_window = (np.ones(n_windows) * indv).astype(int)
        if len(X_test) == 0:
            X_test = idata
            y_test = ilabels
            I_test  = I_window
        else:
            X_test = np.concatenate([X_test, idata])
            y_test = np.concatenate([y_test, ilabels])
            I_test = np.concatenate([I_test, I_window])

    X_val = []
    y_val = []
    I_val = []
    for indv in val_ids:
        idata, ilabels = get_activity_values(indv, window_size, data_map)
        n_windows = len(idata) # number of windows for this ind
        I_window = (np.ones(n_windows) * indv).astype(int)
        if len(X_val) == 0:
            X_val = idata
            y_val = ilabels
            I_val  = I_window
        else:
            X_val = np.concatenate([X_val, idata])
            y_val = np.concatenate([y_val, ilabels])
            I_val = np.concatenate([I_val, I_window])
    
    I_train = np.array(I_train)
    I_test = np.array(I_test)
    I_val = np.array(I_val)

    
    data_map =  {
            'train': [train_ids, X_train, y_train, I_train],
            'test': [test_ids, X_test, y_test, I_test],
            'val':[val_ids, X_val, y_val, I_val]
    }
    
    np.save(DB_CACHE_PATH, data_map)
    
    return data_map

# Modes: leave-one-subject, shuffle-all, custom
class DatasetOpenDataset:
    def __init__(self, mode, seconds = 4, folds = []):
        self.mode = mode
        self.signals = np.array(columns)
        self.activities = activities
        self.activities_map = activities_map
        self.users = individuals.tolist()
        self.intensity_map = activities_met_map
        self.intensities = [
            "Sedentary",
            "Light"
        ]        
        self.currFold = -1
        self.N_TESTS = 1
        self.seconds = seconds
        self.folds = folds # List of test ids lists 
    
    def filterSignals(self, signals):
        signals = np.array(signals)
        idx = [np.where(self.signals == dim)[0][0] for dim in signals]
        self.X_train = self.X_train[:, :, idx]
        self.X_test = self.X_test[:, :, idx]
    
    def loadData(self, activities = None):
        if self.mode == 'leave-one-subject':
            self.currFold = self.currFold + 1
            if self.currFold == len(self.users):
                return False
            self.test_users = self.users[self.currFold: self.currFold + self.N_TESTS]
            self.train_users = self.users[:self.currFold] + self.users[self.currFold + self.N_TESTS:]
        elif self.mode == 'custom':
            self.currFold = self.currFold + 1
            if self.currFold == len(self.folds):
                return False
            self.test_users = self.folds[self.currFold]
            self.train_users = [x for x in self.users if x not in self.test_users]
        else:
            self.train_users = self.users
            self.test_users = []
        
        
        
        data = read_open_dataset(
            seconds=self.seconds,
            train_ids=self.train_users, 
            test_ids=self.test_users, 
            cache=True
        )
        
        if self.mode == 'leave-one-subject' or self.mode == 'custom':
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
            
        if activities is not None:
            actLabels = [self.activities.index(act) for act in activities]
            
            indices = np.in1d(self.y_train, actLabels)
            self.y_train = self.y_train[indices]
            self.X_train = self.X_train[indices]
            self.I_train = self.I_train[indices]
            self.MET_train = self.MET_train[indices]
            self.Int_train = self.Int_train[indices]
            
            indices = np.in1d(self.y_test, actLabels)
            self.y_test = self.y_test[indices]
            self.X_test = self.X_test[indices]
            self.I_test = self.I_test[indices]
            self.MET_test = self.MET_test[indices]
            self.Int_test = self.Int_test[indices]
        return True


def openDatasetParticipants():
    df = pd.read_csv(os.path.join(DATASET_PATH, 'participants_details.csv'))
    # Removing conflicting participant
    df = df[df.Id != 114]
    return df

