# import torch
import random
import os
import csv
import numpy as np
from math import *
import source.augmentation as aug
from sklearn import svm
from sklearn.ensemble import AdaBoostClassifier
from xgboost import XGBClassifier
import matplotlib.pyplot as plt
from sklearn import metrics
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
import matplotlib.pyplot as plt

import numpy as np
np.set_printoptions(precision=3)

RESULTS_PATH = 'results'

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
        
    


class MetricsSaver:
    def __init__(self, name: str, rows: list, cols: list):
        self.name = name
        self.logs = { row: { col:[] for col in cols} for row in rows}
        self.rows = rows
        self.cols = cols
    
    def addLog(self, row, col, val):
        self.logs[row][col].append(val)
        
    def toImage(self):
        create_dir(RESULTS_PATH)
        path = os.path.join(RESULTS_PATH, 'metrics')
        create_dir(path)
        
        mat_data = []
        
        for row in self.rows:
            vals = list(self.logs[row].values())
            mat_data.append(vals)
        
        for i in range(len(self.rows)):
            for j in range(len(self.cols)):
                vals = np.array(mat_data[i][j])
                vmean = vals.mean()
                vstd = vals.std()
                mat_data[i][j] = '{:.2f} ({:.2f})'.format(vmean, vstd)
                # vmean = vals.mean()
            # vstd = vals.std()
        
        path = os.path.join(path, '{}.png'.format(self.name))
        
        
        plotMatResult(
            self.name, 
            '', 
            self.rows, 
            self.cols, 
            mat_data, 
            plot_fig=True, 
            save_fig=True, 
            file_name=path,
        )
    

def saveConfusionMatrix(real_classes, predicted_classes, name, labels = None, xrotation = 45):
    # cm = confusion_matrix(real_classes, predicted_classes)
    # disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels= CLASS_LABELS)
    # disp.plot()
    
    create_dir(RESULTS_PATH)
    path = os.path.join(RESULTS_PATH, 'confusion_matrix')
    create_dir(path)
    path = os.path.join(path,  'confusion_matrix_{}.png'.format(name))
    
    ConfusionMatrixDisplay.from_predictions (real_classes, predicted_classes, display_labels= labels, xticks_rotation=xrotation)
    
    plt.savefig(path, bbox_inches="tight", dpi=1200)

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

def removeNanWindows(windows):
    new_windows = []
    for window in windows:
        if np.isnan(window).any():
            continue
        else:
            new_windows.append(window)
    return np.array(new_windows)

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

# # Batch of shape BxDxT
# def getRandomSlides(batch, size, isNumpy = False):
#     if not isNumpy:
#         batch = batch.numpy()
#     B, D, T = batch.shape
#     b = np.array([random.randint(0, T - size) for i in range(B)])

#     # slides = np.array([ batch[i,:, b[i]: b[i] + size] for i in range(B)]).astype(np.float32)

#     print("WROHN!!!!!!!!!!!!!!!!!!!!!!!!!!!")
#     slides = []
#     for i in range(B):
#         slide = batch[i,:, b[i]: b[i] + size]
#         for j in range(D):
#             slide[j] = slide[j] + random.uniform(-0.1, 0.1)
#         slides.append(slide)
#     return np.array(slides).astype(np.float32)

#     return slides


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
    


def getColor(pos, unit_range = False):
    colors = [[249, 23, 29], [2, 111, 203], [244, 207, 59], [34, 191, 48], [45, 190, 241], [254, 128, 42], [250, 70, 135]]
    if unit_range:
        colors = [[i / 255.0 for i in c] for c in colors]
    if pos < len(colors):
        return colors[pos]    
    else:
        return list(np.random.choice(range(256), size=3) / (255.0 if unit_range else 1) ) 
    

def classify_dataset(X_train, y_train, X_test, y_test):
    # clf = RandomForestClassifier(random_state=0)
    # clf = LinearSVC(dual=False, random_state=123)
    # clf = svm.SVC()
    clf = XGBClassifier()

    # clf = AdaBoostClassifier()
    
    # clf = KNeighborsClassifier()
    clf.fit(X_train, y_train)
    
    
    return clf.predict(X_train), clf.predict(X_test)
    
# X -> (N, T, D)
def idsMean(ids, X, I):
    D = X.shape[2]
    ind_mean = []
    for ind in ids:
        X_indices = np.where(I==ind)
        means = []
        for k in range(D):
            dmean = np.mean( X[X_indices][:, :, k])
            means.append(dmean)
        ind_mean.append(means)
    ind_mean = np.array(ind_mean)
    return ind_mean

# X -> (N, T, D)
def idsStd(ids, X, I):
    D = X.shape[2]
    ind_std = []
    for ind in ids:
        X_indices = np.where(I==ind)[0]
        stdss = []
        for k in range(D):
            dstd = np.std( X[X_indices][:, :, k])
            stdss.append(dstd)
        ind_std.append(stdss)
    ind_std = np.array(ind_std)
    return ind_std


def plot1d(x, x2=None, x3=None, ylim=(-1, 1), save_file=""):
    import matplotlib.pyplot as plt

    plt.figure(figsize=(6, 3))
    steps = np.arange(x.shape[0])
    plt.plot(steps, x)
    if x2 is not None:
        plt.plot(steps, x2)
    if x3 is not None:
        plt.plot(steps, x3)
    plt.xlim(0, x.shape[0])
    # plt.ylim(ylim)
    plt.tight_layout()
    if save_file:
        plt.savefig(save_file, "")
    else:
        plt.show()
    return

def rescale(val, in_min, in_max, out_min, out_max):
    return out_min + (val - in_min) * ((out_max - out_min) / (in_max - in_min))

# MinMax normalization for univariate time series
def normal_normalization(windows, minv=None, maxv=None):
    norm_windows = np.copy(windows)

    if minv is None:
        min_val = np.min(norm_windows)
        max_val = np.max(norm_windows)
    else:
        min_val = minv
        max_val = maxv

    norm_windows = (norm_windows - min_val) / (max_val - min_val) 
    # norm_windows = np.array([ [rescale(i, min_val, max_val, 0, 1) for i in e]  for e in norm_windows]) 

    return norm_windows, min_val, max_val


# MinMax normalization for multivariate time series
def mts_norm(X, minl = [], maxl= []):
    norm_X = X.transpose([0, 2, 1])
    N, D, T = norm_X.shape
    min_l = []
    max_l = []
    for d in range(D):
        if len(minl) == 0:
            norm_windows, minv, maxv = normal_normalization(norm_X[:,d,:])
        else:
            norm_windows, minv, maxv = normal_normalization(norm_X[:,d,:], minv=minl[d], maxv=maxl[d])
        min_l.append(minv)
        max_l.append(maxv)
        norm_X[:,d, :] = norm_windows
    return norm_X.transpose([0, 2, 1]), min_l, max_l


def plotMatResult(title_text, footer_text, row_names, col_names, mat_data, file_name = 'temp.png', save_fig = False, plot_fig=True):
    fig_background_color = 'white'
    fig_border = 'steelblue'
    
    data = []
    for r in range(len(mat_data)):
        data.append([row_names[r]] + list(mat_data[r]))
    data.insert(0, col_names)
    
    # Pop the headers from the data array
    column_headers = data.pop(0)
    row_headers = [x.pop(0) for x in data]# Table data needs to be non-numeric text. Format the data
    # while I'm at it.
    cell_text = []
    for row in data:
        # cell_text.append([f'{x/1000:1.1f}' for x in row])# Get some lists of color specs for row and column headers
        cell_text.append([str(x) for x in row])# Get some lists of color specs for row and column headers
    rcolors = plt.cm.BuPu(np.full(len(row_headers), 0.1))
    ccolors = plt.cm.BuPu(np.full(len(column_headers), 0.1))# Create the figure. Setting a small pad on tight_layout
    # seems to better regulate white space. Sometimes experimenting
    # with an explicit figsize here can produce better outcome.
    plt.figure(linewidth=2,
            edgecolor=fig_border,
            facecolor=fig_background_color,
            tight_layout={'pad':1},
            #figsize=(5,3)
            )# Add a table at the bottom of the axes
    the_table = plt.table(cellText=cell_text,
                        rowLabels=row_headers,
                        rowColours=rcolors,
                        rowLoc='right',
                        colColours=ccolors,
                        colLabels=column_headers,
                        loc='center')# Scaling is the only influence we have over top and bottom cell padding.
    # Make the rows taller (i.e., make cell y scale larger).
    the_table.scale(1, 1.5)# Hide axes
    ax = plt.gca()
    ax.get_xaxis().set_visible(False)
    ax.get_yaxis().set_visible(False)# Hide axes border
    plt.box(on=None)# Add title
    plt.suptitle(title_text)# Add footer
    plt.figtext(0.95, 0.05, footer_text, horizontalalignment='right', size=8, weight='light')# Force the figure to update, so backends center objects correctly within the figure.
    # Without plt.draw() here, the title will center on the axes and not the figure.
    plt.draw()# Create image. plt.savefig ignores figure edge and face colors, so map them.
    fig = plt.gcf()
    
    if save_fig:
        plt.savefig(file_name,
            #bbox='tight',
            edgecolor=fig.get_edgecolor(),
            facecolor=fig.get_facecolor(),
            dpi=300
        )
    if plot_fig:
        plt.show()
    