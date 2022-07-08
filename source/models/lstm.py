# import tensorflow as tf
from keras.models import Sequential
from keras.layers import LSTM, Conv1D
from keras.layers import Dense
from keras.layers import RepeatVector
from keras.layers import TimeDistributed
from keras.optimizers import SGD
from keras import Model
from keras import backend as K
from keras.metrics import CategoricalAccuracy
from keras.regularizers import l2

lstm_embedding = 64

def lstmModel(timesteps, n_features):
    opt = SGD(lr=0.01, momentum=0.9, clipvalue=0.5)
    model = Sequential()
    model.add(LSTM(128, activation='relu', input_shape=(timesteps,n_features), return_sequences=True))
    model.add(LSTM(lstm_embedding, activation='relu', return_sequences=False))
    model.add(RepeatVector(timesteps))
    model.add(LSTM(64, activation='relu', return_sequences=True))
    model.add(LSTM(128, activation='relu', return_sequences=True))
    model.add(TimeDistributed(Dense(n_features, activation='sigmoid')))
    # model.compile(optimizer='adam', loss='mse')
    # model.compile(optimizer='adam', loss='binary_crossentropy')
    model.compile(optimizer=opt, loss='binary_crossentropy')
    
    
    # model.summary()
    encoder = Model(model.input, model.get_layer(index = 1).output )
    return model, encoder


def lstmModel_classification(timesteps, n_features, C):
    opt = SGD(lr=0.01, momentum=0.9, clipvalue=0.5)
    model = Sequential()
    model.add(Conv1D(8, 5, strides =1 , padding='same', activation='relu', input_shape=(timesteps,n_features)))
    model.add(Conv1D(16, 5, strides =1 , padding='same', activation='relu'))
    model.add(Conv1D(32, 5, strides =1 , padding='same', activation='relu'))
    model.add(Conv1D(64, 5, strides =1 , padding='same', activation='relu'))
    model.add(Conv1D(128, 5, strides =1 , padding='same', activation='relu'))
    model.add(Conv1D(256, 5, strides =1 , padding='same', activation='relu'))
    model.add(LSTM(128, dropout=0.1, return_sequences=True))
    model.add(LSTM(128, dropout=0.1))
    model.add(Dense(64))
    # model.add(LSTM(90))
    # model.add(LSTM(256, activation='relu', input_shape=(timesteps,n_features), return_sequences=True))
    # model.add(LSTM(lstm_embedding, activation='relu', return_sequences=False))
    # model.add(LSTM(256, activation='relu', return_sequences=False, dropout=0.2))
    # model.add(LSTM(256, activation='sigmoid', return_sequences=True, dropout=0.2))
    # model.add(RepeatVector(timesteps))
    # model.add(LSTM(64, activation='relu', return_sequences=False))
    # model.add(LSTM(256, activation='sigmoid', return_sequences=False))
    # model.add(TimeDistributed(Dense(n_features, activation='sigmoid')))
    # model.add(Dense(units = 256, activation='sigmoid'))
    # model.add(Dense(units = 64, activation='sigmoid'))
    model.add(Dense(units = C, activation='softmax'))
    model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=[CategoricalAccuracy()])
    
    
    
    model.summary()
    
    # model.pr
    # model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
    # model.compile(optimizer=opt, loss='binary_crossentropy')
    # return 0
    return model
    # model.summary()
    # encoder = Model(model.input, model.get_layer(index = 1).output )
    # return model, encoder
    
    
def lstmModel_prediction(timesteps, n_features):
    opt = SGD(lr=0.01, momentum=0.9, clipvalue=0.5)
    model = Sequential()
    model.add(Conv1D(8, 5, strides =1 , padding='same', activation='relu', input_shape=(timesteps,n_features)))
    model.add(Conv1D(16, 5, strides =1 , padding='same', activation='relu', kernel_regularizer=l2(0.01), bias_regularizer=l2(0.01)))
    model.add(Conv1D(32, 5, strides =1 , padding='same', activation='relu', kernel_regularizer=l2(0.01), bias_regularizer=l2(0.01)))
    model.add(Conv1D(64, 5, strides =1 , padding='same', activation='relu', kernel_regularizer=l2(0.01), bias_regularizer=l2(0.01)))
    model.add(Conv1D(128, 5, strides =1 , padding='same', activation='relu', kernel_regularizer=l2(0.01), bias_regularizer=l2(0.01)))
    model.add(Conv1D(256, 5, strides =1 , padding='same', activation='relu', kernel_regularizer=l2(0.01), bias_regularizer=l2(0.01)))
    model.add(LSTM(128, dropout=0.1, return_sequences=True, kernel_regularizer=l2(0.01), recurrent_regularizer=l2(0.01), bias_regularizer=l2(0.01)))
    model.add(LSTM(128, dropout=0.1, kernel_regularizer=l2(0.01), recurrent_regularizer=l2(0.01), bias_regularizer=l2(0.01)))
    model.add(Dense(64, kernel_regularizer=l2(0.01), bias_regularizer=l2(0.01)))
    # model.add(LSTM(90))
    # model.add(LSTM(256, activation='relu', input_shape=(timesteps,n_features), return_sequences=True))
    # model.add(LSTM(lstm_embedding, activation='relu', return_sequences=False))
    # model.add(LSTM(256, activation='relu', return_sequences=False, dropout=0.2))
    # model.add(LSTM(256, activation='sigmoid', return_sequences=True, dropout=0.2))
    # model.add(RepeatVector(timesteps))
    # model.add(LSTM(64, activation='relu', return_sequences=False))
    # model.add(LSTM(256, activation='sigmoid', return_sequences=False))
    # model.add(TimeDistributed(Dense(n_features, activation='sigmoid')))
    # model.add(Dense(units = 256, activation='sigmoid'))
    # model.add(Dense(units = 64, activation='sigmoid'))
    model.add(Dense(units = 1, activation='relu', kernel_regularizer=l2(0.01), bias_regularizer=l2(0.01)))
    model.compile(optimizer='adam', loss='mse')
    
    
    
    model.summary()
    
    # model.pr
    # model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
    # model.compile(optimizer=opt, loss='binary_crossentropy')
    # return 0
    return model
    # model.summary()
    # encoder = Model(model.input, model.get_layer(index = 1).output )
    # return model, encoder

