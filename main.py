import pandas as pd
import numpy as np
import time

from sklearn import preprocessing

from keras.models import Sequential, load_model
from keras.layers import Dense, Dropout
from keras.optimizers import Adam, RMSprop
from keras.callbacks import ModelCheckpoint, EarlyStopping

import matplotlib.pyplot as plt

def get_dataset():
    # 訓練集
    data_train = pd.read_csv('./train/train-v3.csv')
    x_train = data_train.drop(['id', 'price'], axis=1).values   # 房屋資訊
    y_train = data_train['price'].values    # 房價
    y_train = y_train.reshape((-1, 1))  # 列轉行

    # 測試集
    data_valid = pd.read_csv('./train/valid-v3.csv')
    x_valid = data_valid.drop(['id', 'price'], axis=1).values   # 房屋資訊
    y_valid = data_valid['price'].values    # 房價
    y_valid = y_valid.reshape((-1, 1))  # 列轉行

    # 數據正規化
    x_train = preprocessing.scale(x_train)
    x_valid = preprocessing.scale(x_valid)

    # 繪製柱狀圖
    # plt.figure()
    # plt.subplot(1, 2, 1)
    # plt.hist(y_train)
    # plt.subplot(1, 2, 2)
    # plt.hist(np.log1p(y_train))

    # for i in range(len(x_train)-1):
    #     data = x_train[:,i]
    #     print(x_train[:,i])
    #     plt.hist(data)
    #     plt.show()
    
    return  x_train, y_train, x_valid, y_valid

def create_model(input_shape):
    model = Sequential()
    model.add(Dense(units=32, activation='relu', input_shape=input_shape))
    model.add(Dense(units=64, activation='relu'))
    model.add(Dense(units=32, activation='relu'))
    model.add(Dense(units=1))   # 輸出層

    print(model.summary())
    
    return model

def create_model_1(input_shape):
    model = Sequential()
    model.add(Dense(units=64, activation='relu', input_shape=input_shape))
    model.add(Dense(units=64, activation='relu'))
    model.add(Dense(units=128, activation='relu'))
    model.add(Dense(units=1))   # 輸出層

    print(model.summary())
    
    return model

def create_model_2(input_shape):
    model = Sequential()
    model.add(Dense(units=32, activation='relu', input_shape=input_shape))
    model.add(Dense(units=32, activation='relu'))
    model.add(Dense(units=64, activation='relu'))
    model.add(Dense(units=64, activation='relu'))
    model.add(Dense(units=128, activation='relu'))
    model.add(Dense(units=128, activation='relu'))
    model.add(Dense(units=1))   # 輸出層

    print(model.summary())
    
    return model

def training_model(model, x_train, y_train, x_valid, y_valid, batch_size, epochs, stop_epochs):
    PATH = 'model_%s.hdf5' % time.strftime('%y%m%d')
    ckpt = ModelCheckpoint(PATH, monitor='val_MAE', verbose=1, save_best_only=True, mode='min')
    early_stopping = EarlyStopping(monitor='val_MAE', patience=stop_epochs)
    
    history = model.fit(x=x_train, y=y_train, batch_size=batch_size, epochs=epochs, verbose=1, validation_data=(x_valid, y_valid), callbacks=[ckpt, early_stopping])
    # print(history.history.keys())

    [loss, mae] = model.evaluate(x_valid, y_valid, verbose=0)
    print("Testing set Mean Abs Error: ${:7.2f}".format(mae))

    test_predictions = model.predict(x_valid).flatten()
    test_answer = y_valid.flatten()

    plt.figure(1)
    plt.subplot(2, 2, 1)
    plt.scatter(test_answer, test_predictions)
    plt.xlabel('True Values')
    plt.ylabel('Predictions')
    plt.axis('equal')
    plt.xlim(plt.xlim())
    plt.ylim(plt.ylim())

    plt.subplot(2, 2, 2)
    error = test_predictions - test_answer
    plt.hist(error, bins=50)
    plt.xlabel("Prediction Error")

    plt.figure(1)
    plt.subplot(2, 2, 3)
    plt.xlabel('Epoch')
    plt.ylabel('MAE')
    plt.plot(history.epoch, np.array(history.history['MAE']), label='Train MAE')
    plt.plot(history.epoch, np.array(history.history['val_MAE']), label='Valid MAE')
    plt.legend()

    plt.subplot(2, 2, 4)
    plt.xlabel('Epoch')
    plt.ylabel('loss')
    plt.plot(history.epoch, np.array(history.history['loss']), label='Train loss')
    plt.plot(history.epoch, np.array(history.history['val_loss']), label='Valid loss')
    plt.legend()

    plt.show()

    return model
    
if __name__ == '__main__':
    # # Training
    # batch_size = 32
    # epochs = 2000
    # stop_epochs = 100

    # x_train, y_train, x_valid, y_valid = get_dataset()
    # # model = create_model(x_train.shape[1:])
    # # model = create_model_1(x_train.shape[1:])
    # model = create_model_2(x_train.shape[1:])
    # optimizer = RMSprop(learning_rate=0.001)
    # model.compile(optimizer=optimizer, loss='MAE', metrics=['MAE'])
    # model = training_model(model, x_train, y_train, x_valid, y_valid, batch_size, epochs, stop_epochs)
    
    # model_191108_1:b=32, e=2000, s=100, opt=RMSprop(learning_rate=0.001)create_model
    # model_191108_2:b=64, e=2000, s=100, opt=RMSprop(learning_rate=0.001)create_model_1
    # model_191108_3:b=64, e=2000, s=100, opt=RMSprop(learning_rate=0.001)create_model_2
    # model_191108_4:b=32, e=2000, s=100, opt=RMSprop(learning_rate=0.001)create_model_2

    # Predict
    data_test = pd.read_csv('./test/test-v3.csv')
    x_test = data_test.drop(['id'], axis=1).values
    x_test = preprocessing.scale(x_test)    # 數據正規化

    # model = load_model('./model_191108_4.hdf5')
    # y_predict = model.predict(x_test, verbose=1).flatten() # 攤平陣列
    # pd.DataFrame({"id": list(range(1, len(y_predict)+1)), "price": y_predict}).to_csv('output.csv', index=False, header=True)

    model_1 = load_model('./model_191108_2.hdf5')
    model_2 = load_model('./model_191108_4.hdf5')
    y_predict_1 = model_1.predict(x_test, verbose=1).flatten()
    y_predict_2 = model_2.predict(x_test, verbose=1).flatten()
    y_predict = (y_predict_1+y_predict_2) / 2
    pd.DataFrame({"id": list(range(1, len(y_predict)+1)), "price": y_predict}).to_csv('output.csv', index=False, header=True)