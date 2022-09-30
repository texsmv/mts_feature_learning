from source.read_HAR_dataset import read_har_dataset, har_dimensions, har_activities, har_activities_map, har_ind_IDS
from source.utils import  filter_dimensions
from source.tserie import TSerie
from source.utils import classify_dataset

import umap



DATASET = 'HAR-UML20'
KFOLDS = 10
N_TESTS = 2


for k in range(KFOLDS):
    if DATASET == 'HAR-UML20':
        all_ids = har_ind_IDS
        test_ids = all_ids[k: k + N_TESTS]
        train_ids = all_ids[:k] + all_ids[k + N_TESTS:]        
        
        data = read_har_dataset('./datasets/HAR-UML20/', train_ids=train_ids, test_ids=test_ids, val_ids=[])
        ids_train, X_train, y_train, I_train, train_kcal_MET = data['train']
        # ids_val, X_val, y_val, I_val, val_kcal_MET = data['val']
        ids_test, X_test, y_test, I_test, test_kcal_MET = data['test']
        
        all_dimensions = har_dimensions
        activities_map = har_activities_map
        dimensions = ['Accelerometer-X', 'Accelerometer-Y', 'Accelerometer-Z', 'Gyrometer-X', 'Gyrometer-Y', 'Gyrometer-Z']


    X_train_f = filter_dimensions(X_train, har_dimensions, dimensions)
    X_test_f = filter_dimensions(X_test, har_dimensions, dimensions)

    mts_train = TSerie(X = X_train_f, y = y_train, I = I_train, dimensions = dimensions, classLabels=activities_map)
    mts_test = TSerie(X = X_test_f, y = y_test, I = I_test, dimensions = dimensions, classLabels=activities_map)
    
    mts_train.folding_features_v2()
    mts_test.folding_features_v2()
    
    reducer = umap.UMAP(n_components=32, metric='braycurtis')
    embeddings_train = reducer.fit_transform(mts_train.features, y = mts.y)
    embeddings_test = reducer.transform(mts_test.features)
    
    clf = AdaBoostClassifier()
    clf.fit(X_train, y_train)
    pred_train, pred_test = clf.predict(X_train), clf.predict(X_test)
    