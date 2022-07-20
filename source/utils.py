import torch
import random
import os
import csv
import numpy as np
from math import *
import source.augmentation as aug

from sklearn import metrics

import numpy as np
np.set_printoptions(precision=3)

class ValueLogger(object):
    """Computes and stores the average and current value"""
    def __init__(self, name, epoch_freq = 5):
        self.name = name
        self.epoch_freq = epoch_freq
        self.reset()
    
    def reset(self):
        self.avgs = []
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0.0
        self.bestAvg = np.inf
        
        
    def end_epoch(self):
        
        self.avgs = self.avgs + [self.avg]
        self.val = 0
        self.sum = 0
        self.count = 0.0
        if len(self.avgs) == 1 or len(self.avgs) % self.epoch_freq == 0:
            print("Epoch[{}] {} {}: {}".format(len(self.avgs), self.name, "avg", self.avg))
    
        if self.bestAvg > self.avg:
            self.bestAvg = self.avg
            return True
        else:
            return False

    # Updates de value history
    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count




def regression_results(y_true, y_pred):
    
    # Regression metrics
    explained_variance=metrics.explained_variance_score(y_true, y_pred)
    mean_absolute_error=metrics.mean_absolute_error(y_true, y_pred) 
    # mse=metrics.mean_squared_error(y_true, y_pred) 
    # mean_squared_log_error=metrics.mean_squared_log_error(y_true, y_pred)
    # median_absolute_error=metrics.median_absolute_error(y_true, y_pred)
    r2=metrics.r2_score(y_true, y_pred)
    
    
    return round(mean_absolute_error,4), round(explained_variance,4)
    return round(mean_absolute_error,4), round(r2,4)

    # print('explained_variance: ', round(explained_variance,4))    
    # print('mean_squared_log_error: ', round(mean_squared_log_error,4))
    # print('r2: ', round(r2,4))
    # print('MAE: ', round(mean_absolute_error,4))
    # print('MSE: ', round(mse,4))
    # print('RMSE: ', round(np.sqrt(mse),4))

def smoothAvg(serie, ksize):
    ksize = 10
    kernel = np.ones(ksize) / ksize
    return np.convolve(serie, kernel, mode='same')

def splitWindows(vector, wsize):
    n = floor(len(vector) / wsize)
    return np.array([vector[i * wsize: (i + 1) * wsize] for i in range(n)])


# splitOverlapWindows
# Split the windows with an overlap of 50%
def splitOverlapWindows(vector, wsize, overlap = 0.5):
    wadv = int(wsize * (1 - overlap))
    wovep = wsize - wadv
    n = floor((len(vector) - wovep) / (wadv))
    # print(wadv)
        
    return np.array([vector[i * wadv: i * wadv + wsize] for i in range(n)])

def divide_train_test(X, n_test, Z = None):
    """
    Divides the users in the individuals list into a test and train set.
    The test set is the n_test randomly chossen users.
    Can divide a second list as well.
    """
    X = np.array(X)
    if Z is not None:
        print(Z)
        Z = np.array(Z)
    
    idx = np.random.choice(np.arange(len(X)), n_test, replace=False).astype(int)
    X_test = X[idx]
    X_train = np.delete(X, idx, axis=0)

    if Z is None:
        return X_train, X_test

    Z_test = Z[idx]
    Z_train = np.delete(Z, idx, axis=0)

    return X_train, X_test, Z_train, Z_test



def create_dir(path):
    # Check whether the specified path exist or not
    isExist = os.path.exists(path)
    if not isExist:
      # Create a new directory because it does not exist 
      os.makedirs(path)
      print("The new directory is created!")


# Assuming X has the shape N, T, D
def filter_dimensions(X, all_dimensions, dimensions):
    idx = [np.where(all_dimensions == dim)[0][0] for dim in dimensions]
    return X[:, :, idx]



def save_metrics3(labels_map, path):
    with open(path, 'w', newline='') as csvfile:
        spamwriter = csv.writer(csvfile, delimiter=',',
                                quoting=csv.QUOTE_MINIMAL)
        spamwriter.writerow( ['sensors', 'accuracy_tr', 'balanced_accuracy_tr', 'f1_tr', 'accuracy_te', 'balanced_accuracy_te', 'f1_te', 'accuracy_val', 'balanced_accuracy_val', 'f1_val'])
        for key, value in labels_map.items():
            y_train, train_pred, y_test, test_pred, y_validation, validation_pred = value
            accuracy_tr = metrics.accuracy_score(y_train, train_pred)
            accuracy_te = metrics.accuracy_score(y_test, test_pred)
            accuracy_val = metrics.accuracy_score(y_validation, validation_pred)
            
            bal_accuracy_tr = metrics.balanced_accuracy_score(y_train, train_pred)
            bal_accuracy_te = metrics.balanced_accuracy_score(y_test, test_pred)
            bal_accuracy_val = metrics.balanced_accuracy_score(y_validation, validation_pred)
            
            f1_tr = metrics.f1_score(y_train, train_pred, average='weighted')
            f1_te = metrics.f1_score(y_test, test_pred, average='weighted')
            f1_val = metrics.f1_score(y_validation, validation_pred, average='weighted')
            
            spamwriter.writerow([key, accuracy_tr, bal_accuracy_tr, f1_tr, accuracy_te, bal_accuracy_te, f1_te, accuracy_val, bal_accuracy_val, f1_val])


def save_prediction_metrics(pred_map, path, class_ids, class_labels, train_classes, test_classes):
    with open(path, 'w', newline='') as csvfile:
        spamwriter = csv.writer(csvfile, delimiter=',',
                                quoting=csv.QUOTE_MINIMAL)
        columns_labels = ['sensors', 'all_train_MAE', 'all_train_R2', 'all_test_MAE', 'all_test_R2']
        for l in class_labels:
            # columns_labels =  columns_labels + ['{}_train_MAE'.format(l), '{}_train_R2'.format(l), '{}_test_MAE'.format(l), '{}_test_R2'.format(l)]
            columns_labels =  columns_labels + ['{}_MAE'.format(l)]
        spamwriter.writerow( columns_labels)
        for key, value in pred_map.items():
            row_vals = [key]
            y_train, train_pred, y_test, test_pred, y_val, val_pred = value
            mae_tr, r2_tr = regression_results(y_train, train_pred)
            mae_te, r2_te = regression_results(y_test, test_pred)
            row_vals = row_vals + [mae_tr, r2_tr, mae_te, r2_te]
            
            for i in range(len(class_ids)):
                id = class_ids[i]
                label = class_labels[i]
                
                indices_tr = np.where(train_classes == id)[0]
                
                y_class_tr = y_train[indices_tr]
                class_pred_tr = train_pred[indices_tr]
                mae_tr, r2_tr = regression_results(y_class_tr, class_pred_tr)
                
                indices_te = np.where(test_classes == id)[0]
                
                y_class_te = y_test[indices_te]
                class_pred_te = test_pred[indices_te]
                mae_te, r2_te = regression_results(y_class_te, class_pred_te)
                
                # row_vals = row_vals + [mae_tr, r2_tr, mae_te, r2_te]
                row_vals = row_vals + [mae_te]
            spamwriter.writerow( row_vals)
            
    

def save_metrics2(labels_map, path):
    with open(path, 'w', newline='') as csvfile:
        spamwriter = csv.writer(csvfile, delimiter=',',
                                quoting=csv.QUOTE_MINIMAL)
        spamwriter.writerow( ['sensors', 'accuracy_tr', 'balanced_accuracy_tr', 'f1_tr', 'accuracy_te', 'balanced_accuracy_te', 'f1_te'])
        for key, value in labels_map.items():
            y_train, train_pred, y_test, test_pred = value
            accuracy_tr = metrics.accuracy_score(y_train, train_pred)
            accuracy_te = metrics.accuracy_score(y_test, test_pred)
            # accuracy_val = metrics.accuracy_score(y_validation, validation_pred)
            
            bal_accuracy_tr = metrics.balanced_accuracy_score(y_train, train_pred)
            bal_accuracy_te = metrics.balanced_accuracy_score(y_test, test_pred)
            # bal_accuracy_val = metrics.balanced_accuracy_score(y_validation, validation_pred)
            
            f1_tr = metrics.f1_score(y_train, train_pred, average='weighted')
            f1_te = metrics.f1_score(y_test, test_pred, average='weighted')
            # f1_val = metrics.f1_score(y_validation, validation_pred, average='weighted')
            
            spamwriter.writerow([key, accuracy_tr, bal_accuracy_tr, f1_tr, accuracy_te, bal_accuracy_te, f1_te])

def create_dir(path):
    # Check whether the specified path exist or not
    isExist = os.path.exists(path)
    if not isExist:
      # Create a new directory because it does not exist 
      os.makedirs(path)
      print("The new directory is created!")

# Batch of shape BxDxT
def getRandomSlides(batch, size, isNumpy = False):
    if not isNumpy:
        batch = batch.numpy()
    B, D, T = batch.shape
    b = np.array([random.randint(0, T - size) for i in range(B)])

    slides = np.array([ batch[i,:, b[i]: b[i] + size] for i in range(B)]).astype(np.float32)

    # slides = []
    # for i in range(B):
    #     slide = batch[i,:, b[i]: b[i] + size]
    #     for j in range(D):
    #         slide[j] = slide[j] + random.uniform(-0.1, 0.1)
    #     slides.append(slide)
    # return np.array(slides).astype(np.float32)

    return slides


def getViews(batch, size, isNumpy = False):
    originalSlides = getRandomSlides(batch, size, isNumpy)
    
    fslides = originalSlides.transpose((0, 2, 1))
    # scaled = aug.scaling(fslides, sigma=0.1).transpose((0, 2, 1))
    scaled = fslides.transpose((0, 2, 1))
    
    originalSlides = getRandomSlides(batch, size, isNumpy)
    fslides = originalSlides.transpose((0, 2, 1))
    # flipped = aug.rotation(fslides).transpose((0, 2, 1))
    flipped = fslides.transpose((0, 2, 1))
    
    originalSlides = getRandomSlides(batch, size, isNumpy)
    fslides = originalSlides.transpose((0, 2, 1))
    # magWarped =  aug.magnitude_warp(fslides, sigma=0.2, knot=4).transpose((0, 2, 1))
    magWarped =  fslides.transpose((0, 2, 1))
    
    originalSlides = getRandomSlides(batch, size, isNumpy)
    # fslides = originalSlides.transpose((0, 2, 1))
    
    # return torch.from_numpy(originalSlides.astype(np.float32))
    return [
        torch.from_numpy(originalSlides.astype(np.float32)),
        torch.from_numpy(scaled.astype(np.float32)),
        torch.from_numpy(flipped.astype(np.float32)),
        torch.from_numpy(magWarped.astype(np.float32)),
    ]
    # return torch.from_numpy(np.stack([originalSlides, scaled, flipped, magWarped], axis=1).astype(np.float32))