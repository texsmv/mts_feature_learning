import torch
from torch import nn

import torch.nn.functional as F
from torch.utils.data import TensorDataset
from torch.utils.data import Dataset
from torch.utils.data import DataLoader

from source.utils import getRandomSlides, getViews


class CNNFeatures(nn.Module):
    def __init__(self, in_features, use_batch_norm = True):
        super().__init__()
        k = 5
        p = k // 2
        self.c1 = nn.Conv1d(in_features, 16, k, stride = 1, padding_mode='replicate', padding = p)
        self.c2 = nn.Conv1d(16, 16, k, stride = 1, padding_mode='replicate', padding = p)
        self.c3 = nn.Conv1d(16, 16, k, stride = 1, padding_mode='replicate', padding = p)
        
        self.use_batch_norm = use_batch_norm
        
        if self.use_batch_norm:
            self.bn1 = nn.BatchNorm1d(16)
            self.bn2 = nn.BatchNorm1d(16)
            self.bn3 = nn.BatchNorm1d(16)
        else:
            self.dropout = nn.Dropout(p=0.2)
        
        self.m = nn.MaxPool1d(2)
        
    def forward(self,x):
        x = self.c1(x)
        x = F.relu(x)
        if self.use_batch_norm:
            x = self.bn1(x)
        else:
            x = self.dropout(x)
        x = self.m(x)
        
        
        x = self.c2(x)
        x = F.relu(x)
        if self.use_batch_norm:
            x = self.bn2(x)
        else:
            x = self.dropout(x)
        x = self.m(x)
        
        x = self.c3(x)
        x = F.relu(x)
        if self.use_batch_norm:
            x = self.bn3(x)
        else:
            x = self.dropout(x)
        
        x = torch.flatten(x, start_dim=1)
        x = F.normalize(x, dim=1)
        return x

class HeadModel(nn.Module):
    # head either mlp or linear
    def __init__(self, dim_in, head='mlp',  feat_dim=128):
        super(HeadModel, self).__init__()
        if head == 'linear':
            self.head = nn.Linear(dim_in, feat_dim)
        elif head == 'mlp':
            self.head = nn.Sequential(
                nn.Linear(dim_in, dim_in),
                nn.ReLU(inplace=True),
                nn.BatchNorm1d(dim_in),
                nn.Linear(dim_in, feat_dim)
            )
        else:
            raise NotImplementedError(
                'head not supported: {}'.format(head))

    def forward(self, x):
        x = self.head(x)
        x = F.normalize(x, dim=1)
        return x


class SiameseNetwork(nn.Module):
    def __init__(self, in_features, length, feat_dim=128, head='linear'): # mlp
        super().__init__()
        self.headInFeatures = (length//4) * 16
        self.features = CNNFeatures(in_features)
        self.linear = HeadModel(self.headInFeatures, head = head, feat_dim=feat_dim)
        self.length = length
        self.in_features = in_features
        
    def forward(self,x):
        x = self.features(x)
        x = self.linear(x)
        return x

    def getFeatures(self, x, device, batch_size = 64):
        
        self.features.eval()
        self.linear.eval()
        
        dataset = TensorDataset(torch.from_numpy(x).to(torch.float))
        loader = DataLoader(dataset, batch_size=batch_size)
        
        with torch.no_grad():
            output = []
            for batch in loader:
                x = batch[0]
                x = getRandomSlides(x, self.length)
                x = torch.from_numpy(x)
                x = x.to(device)
                x = self.features(x)

                output.append(x)
                
            output = torch.cat(output, dim=0)
        return output.cpu().numpy()

    def encode(self, x, device, batch_size = 64):
        
        self.features.eval()
        self.linear.eval()
        
        dataset = TensorDataset(torch.from_numpy(x).to(torch.float))
        loader = DataLoader(dataset, batch_size=batch_size)
        
        with torch.no_grad():
            output = []
            for batch in loader:
                x = batch[0]
                x = getRandomSlides(x, self.length)
                x = torch.from_numpy(x)
                x = x.to(device)
                x = self.features(x)
                # x = self.linear(x)

                output.append(x)
                
            output = torch.cat(output, dim=0)
        return output.cpu().numpy()

def train_batch(model, data, optimizer, criterion, device, win_len, supervised= True):
    model.features.train()
    model.linear.train()
    
    optimizer.zero_grad()
    xA, xB, lA, lB = data # Shape BxDxT

    view1, view2, view3, view4 = getViews(xA, win_len)

    B, D, T = xA.shape

    view1 = view1.to(device)
    view2 = view2.to(device)
    view3 = view3.to(device)
    view4 = view4.to(device)
    
    codes1 = model(view1)
    codes2 = model(view2)
    codes3 = model(view3)
    codes4 = model(view4)
    
    viewsCodes = torch.stack([codes1, codes2, codes3, codes4], 1)
    if supervised:
        loss = criterion(viewsCodes, lA)
    else:
        loss = criterion(viewsCodes)
    
    loss.backward()
    optimizer.step()
    return loss.item()
    

def eval_batch(model, data, criterion, device, win_len, supervised= True):
    model.features.eval()
    model.linear.eval()
    
    xA, xB, lA, lB = data

    view1, view2, view3, view4 = getViews(xA, win_len)
    
    view1 = view1.to(device)
    view2 = view2.to(device)
    view3 = view3.to(device)
    view4 = view4.to(device)
    
    codes1 = model(view1)
    codes2 = model(view2)
    codes3 = model(view3)
    codes4 = model(view4)

    B, D, T = xA.shape
    
    viewsCodes = torch.stack([codes1, codes2, codes3, codes4], 1)
    
    if supervised:
        loss = criterion(viewsCodes, lA)
    else:
        loss = criterion(viewsCodes)

    return loss.item()




def train_batch_contrastive(model, data, optimizer, criterion, device, win_len):
    model.features.train()
    model.linear.train()
    
    optimizer.zero_grad()
    xA, xB, lA, lB = data

    anchor_slides = getViews(xA, win_len)[0].to(device)
    positive_slides = getViews(xA, win_len)[0].to(device)
    negative_slides = getViews(xB, win_len)[0].to(device)
    
    B, D, T = xA.shape

    codesAnchor = model(anchor_slides)
    codesPositive = model(positive_slides)
    codesNegative = model(negative_slides)

    positive_labels = torch.zeros(B).to(device)
    negative_labels = torch.ones(B).to(device)

    
    lossPos = criterion(codesAnchor, codesPositive, positive_labels)
    lossNeg = criterion(codesAnchor, codesNegative, negative_labels)
    loss = lossPos + lossNeg * 3

    loss.backward()

    optimizer.step()
    
    return loss.item()
    

def eval_batch_contrastive(model, data, criterion, device, win_len):
    model.features.eval()
    model.linear.eval()
    
    xA, xB, lA, lB = data

    anchor_slides = getViews(xA, win_len)[0].to(device)
    positive_slides = getViews(xA, win_len)[0].to(device)
    negative_slides = getViews(xB, win_len)[0].to(device)
    
    B, D, T = xA.shape

    codesAnchor = model(anchor_slides)
    codesPositive = model(positive_slides)
    codesNegative = model(negative_slides)

    positive_labels = torch.zeros(B).to(device)
    negative_labels = torch.ones(B).to(device)
    
    
    lossPos = criterion(codesAnchor, codesPositive, positive_labels)
    lossNeg = criterion(codesAnchor, codesNegative, negative_labels)
    loss = lossPos + lossNeg * 3

    return loss.item()



def train_batch_triplet(model, data, optimizer, criterion, device, win_len):
    model.features.train()
    model.linear.train()
    
    optimizer.zero_grad()
    xA, xB, lA, lB = data

    anchor_slides = getViews(xA, win_len)[0].to(device)
    positive_slides = getViews(xA, win_len)[0].to(device)
    negative_slides = getViews(xB, win_len)[0].to(device)

    B, D, T = xA.shape

    codesAnchor = model(anchor_slides)
    codesPositive = model(positive_slides)
    codesNegative = model(negative_slides)

    
    loss = criterion(codesAnchor, codesPositive, codesNegative)

    loss.backward()

    optimizer.step()
    
    return loss.item()
    

def eval_batch_triplet(model, data, criterion, device, win_len):
    model.features.eval()
    model.linear.eval()
    
    xA, xB, lA, lB = data

    anchor_slides = getViews(xA, win_len)[0].to(device)
    positive_slides = getViews(xA, win_len)[0].to(device)
    negative_slides = getViews(xB, win_len)[0].to(device)

    B, D, T = xA.shape

    codesAnchor = model(anchor_slides)
    codesPositive = model(positive_slides)
    codesNegative = model(negative_slides)


    loss = criterion(codesAnchor, codesPositive, codesNegative)

    return loss.item()











class CNNVAEFeatures(nn.Module):
    def __init__(self, in_features, out_features, length, device, use_batch_norm = True):
        super().__init__()
        k = 5
        p = k // 2
        self.c1 = nn.Conv1d(in_features, 16, k, stride = 1, padding_mode='replicate', padding = p)
        self.c2 = nn.Conv1d(16, 16, k, stride = 1, padding_mode='replicate', padding = p)
        self.c3 = nn.Conv1d(16, 16, k, stride = 1, padding_mode='replicate', padding = p)
        
        self.cnnFeatures = (length//4) * 16
        
        self.use_batch_norm = use_batch_norm
        
        if self.use_batch_norm:
            self.bn1 = nn.BatchNorm1d(16)
            self.bn2 = nn.BatchNorm1d(16)
            self.bn3 = nn.BatchNorm1d(16)
        else:
            self.dropout = nn.Dropout(p=0.2)
        
        self.m = nn.MaxPool1d(2)
        self.device = device
        
        # * VAE stuff
        self.z_dim = out_features 
        self.linear_mu = nn.Linear(self.cnnFeatures, self.z_dim)
        self.linear_var = nn.Linear(self.cnnFeatures, self.z_dim)
        self.N = torch.distributions.Normal(0, 1)
        # self.N.loc = self.N.loc.cuda() # hack to get sampling on the GPU
        # self.N.scale = self.N.scale.cuda()
        
        self.outLinear = nn.Linear(self.z_dim, out_features)
        
    def forward(self,x):
        B = x.shape[0]
        
        x = self.c1(x)
        x = F.relu(x)
        if self.use_batch_norm:
            x = self.bn1(x)
        else:
            x = self.dropout(x)
        x = self.m(x)
        
        
        x = self.c2(x)
        x = F.relu(x)
        if self.use_batch_norm:
            x = self.bn2(x)
        else:
            x = self.dropout(x)
        x = self.m(x)
        
        x = self.c3(x)
        x = F.relu(x)
        if self.use_batch_norm:
            x = self.bn3(x)
        else:
            x = self.dropout(x)
        
        h = torch.flatten(x, start_dim=1)
        
        h = F.relu(h)
        
        z_mu = self.linear_mu(h)
        
        z_var = self.linear_var(h)
        
        eps = torch.randn(B, self.z_dim).to(self.device)
        
        z_sample = z_mu + torch.exp(z_var / 2) * eps
        # z_sample = z_mu + z_var * self.N.sample(z_mu.shape)
        
        
        
        # x = self.outLinear(z_sample)
        # x = F.normalize(x, dim=1)
        return z_sample, z_mu, z_var

class HeadModelVAE(nn.Module):
    # head either mlp or linear
    def __init__(self, dim_in, head='mlp',  feat_dim=128):
        super(HeadModelVAE, self).__init__()
        if head == 'linear':
            self.head = nn.Linear(dim_in, feat_dim)
        elif head == 'mlp':
            self.head = nn.Sequential(
                nn.Linear(dim_in, dim_in),
                nn.ReLU(inplace=True),
                nn.BatchNorm1d(dim_in),
                nn.Linear(dim_in, feat_dim)
            )
        else:
            raise NotImplementedError(
                'head not supported: {}'.format(head))

    def forward(self, x):
        x = self.head(x)
        x = F.normalize(x, dim=1)
        return x


class VAESiameseNetwork(nn.Module):
    def __init__(self, in_features, length, device, feat_dim=128, head='linear'): # mlp
        super().__init__()
        self.cnn_feat = (length//4) * 16
        
        self.features = CNNVAEFeatures(in_features, feat_dim, length, device)
        
        self.linear = HeadModelVAE(feat_dim, head = head, feat_dim=feat_dim)
        
        self.length = length
        self.in_features = in_features
        
    def forward(self,x):
        
        x, z_mu, z_var = self.features(x)
        c = self.linear(x)
        return c, z_mu, z_var

    def getFeatures(self, x, device, batch_size = 64):
        
        self.features.eval()
        self.linear.eval()
        
        dataset = TensorDataset(torch.from_numpy(x).to(torch.float))
        loader = DataLoader(dataset, batch_size=batch_size)
        
        with torch.no_grad():
            output = []
            for batch in loader:
                x = batch[0]
                x = getRandomSlides(x, self.length)
                x = torch.from_numpy(x)
                x = x.to(device)
                x = self.features(x)

                output.append(x)
                
            output = torch.cat(output, dim=0)
        return output.cpu().numpy()

    def encode(self, x, device, batch_size = 64):
        
        self.features.eval()
        self.linear.eval()
        
        dataset = TensorDataset(torch.from_numpy(x).to(torch.float))
        loader = DataLoader(dataset, batch_size=batch_size)
        
        with torch.no_grad():
            output = []
            for batch in loader:
                x = batch[0]
                x = getRandomSlides(x, self.length)
                x = torch.from_numpy(x)
                x = x.to(device)
                x, _, _ = self.features(x)
                x = self.linear(x)

                output.append(x)
                
            output = torch.cat(output, dim=0)
        return output.cpu().numpy()