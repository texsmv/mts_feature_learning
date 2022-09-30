import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from source.utils import getColor, mts_norm, idsStd

DEBUG = False

class TSerie:
    def __init__(self, X: np.array, y: np.array = None, I: np.array = None, dimensions:np.array =None, classLabels : dict = None):
        self.N = X.shape[0]
        self.T = X.shape[1]
        self.D = X.shape[2]
        self.X = X
        self.y = y
        self.I = I
        self.dimensions = dimensions if dimensions is not None else np.arange(self.D)
        self.features = None
        self.classes = np.unique(y)
        self.individuals = np.unique(I)
        self.classLabels = classLabels if classLabels is not None else {c:str(c) for c in self.classes}
        self.classColors = {self.classes[i]: getColor(i, unit_range=True) for i in range(len(self.classes))}
        self.X_o = np.copy(X)
        if DEBUG:
            print('Loaded mts - N: {}, T: {}, D: {} '.format(self.N, self.T, self.D))
        
    def folding_features_v1(self):
        N, T, D = self.X.shape
        self.features = np.zeros((N, D * T))
        for n in range(N):       
            for d in range(D):
                self.features[n, d * T : (d + 1) * T] = self.X[n, :, d]
        if DEBUG:
            print('Features shape: {}'.format(self.features.shape))

    def folding_features_v2(self):
        N, T, D = self.X.shape
        self.features = np.zeros((N, D * T))
        for n in range(N):
            for t in range(T):
                for d in range(D):
                    self.features[n, D * t + d] = self.X[n, t, d]
        if DEBUG:
            print('Features shape: {}'.format(self.features.shape))
    
    def znorm(self, dimensions=[]):
        if len(dimensions) == 0:
            dimensions = list(range(self.D))
        ids = np.unique(self.I).tolist()
        ind_std = idsStd(ids, self.X, self.I)
        X_norm = np.zeros(self.X.shape)
        
        for i in range(self.N):
            for k in range(self.D):
                if k in dimensions:
                    mag = np.mean(self.X[i, :, k], axis = 0)
                    indice = np.where(ids == self.I[i])[0][0]
                    std = ind_std[indice][k] * 6
                    X_norm[i, :, k] = (self.X[i, :, k] - mag) / std
                else:
                    X_norm[i, :, k] = self.X[i, :, k]
        self.X = X_norm
    
    def center(self, dimensions=[]):
        if len(dimensions) == 0:
            dimensions = list(range(self.D))
        X_norm = np.zeros(self.X.shape)
        
        # magnitudes = self.X.mean(axis=1)
        # magnitudes = np.repeat(magnitudes, self.T, axis=1).reshape([self.N, self.T, self.D])
        
        # stds = np.where(self.y == 0, )
        
        # print(ind_std.shape)
        # print(magnitudes.shape)
        
        for i in range(self.N):
            for k in range(self.D):
                if k in dimensions:
                    mag = np.mean(self.X[i, :, k], axis = 0)
                    X_norm[i, :, k] = (self.X[i, :, k] - mag) 
                else:
                    X_norm[i, :, k] = self.X[i, :, k]
        self.X = X_norm
    # def center():
        
    
    def minMaxNormalization(self, minl=[], maxl=[]):
        if len(minl) == 0:
            self.X, min_l, max_l = mts_norm(self.X)
        else:
            self.X, min_l, max_l = mts_norm(self.X, minl=minl, maxl=maxl)
        return min_l, max_l
    
    def reviewByClass(self, dims, title):
        diff = 1.5 # Distance bettween class bars
        space = 30 # Space between dimension bars
        means = []
        stds = []
        xs = []
        colors = []
        
        fig, axes = plt.subplots(nrows=1, ncols=2, figsize=(10, 4), gridspec_kw={'width_ratios': [3, 1]})
        for i in range(len(self.classes)):
            c = self.classes[i]
            for k in range(len(dims)):
                d = dims[k]
                class_ind = np.where(self.y == c)[0]
                class_data = self.X[class_ind, :, d]
                means.append(np.mean(class_data))
                stds.append(np.std(class_data))
                x = diff * i + space * k
                xs.append(x)
                colors.append(self.classColors[c])
        axes[0].errorbar(xs, means, stds, linestyle='None', marker='o', elinewidth=3, markersize =4, fmt=' ', ecolor =colors,c='black')
        
        axes[1].axis("off") 
        patches = [mpatches.Patch(color=self.classColors[c], label=self.classLabels[c]) for c in self.classes]
        axes[1].legend(handles=patches)
        fig.suptitle(title, fontsize=20)

        plt.show()
    
    def reviewByUser(self, dims, title):
        diff = 1.5 # Distance bettween class bars
        space = 30 # Space between dimension bars
        means = []
        stds = []
        xs = []
        colors = []
        
        iLabels = {c:'participant_'+str(c) for c in self.individuals}
        iColors = {self.individuals[i]: getColor(i, unit_range=True) for i in range(len(self.individuals))}
        
        fig, axes = plt.subplots(nrows=1, ncols=2, figsize=(10, 4), gridspec_kw={'width_ratios': [3, 1]})
        for i in range(len(self.individuals)):
            ind = self.individuals[i]
            for k in range(len(dims)):
                d = dims[k]
                class_ind = np.where(self.I == ind)[0]
                class_data = self.X[class_ind, :, d]
                means.append(np.mean(class_data))
                stds.append(np.std(class_data))
                x = diff * i + space * k
                xs.append(x)
                colors.append(iColors[ind])
        axes[0].errorbar(xs, means, stds, linestyle='None', marker='o', elinewidth=3, markersize =4, fmt=' ', ecolor =colors,c='black')
        axes[1].axis("off") 
        patches = [mpatches.Patch(color=iColors[c], label=iLabels[c]) for c in self.individuals]
        axes[1].legend(handles=patches)
        fig.suptitle(title, fontsize=20)

        plt.show()
    