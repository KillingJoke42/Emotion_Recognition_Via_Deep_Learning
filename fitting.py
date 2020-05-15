# -*- coding: utf-8 -*-
import pickle
import numpy as np
import tensorflow as tf
from tensorflow.keras import models, layers
#from matplotlib import pyplot
import statistics

from sklearn.utils import shuffle
from sklearn import preprocessing

# %% get the dataset

train_data = np.zeros((1,8064,14))
train_labels = np.zeros((1,4))
for i in range(26):
    module = open('D:\Mini II\DATASETS\DEAP\data_preprocessed_python\data_preprocessed_python\s{0:02d}.dat'.format(i+1), 'rb')
    dataset = pickle.load(module, encoding='bytes')
    eegvals = np.array(dataset[b'data'])
    eeglabels = np.array(dataset[b'labels'])
    useful_channels = [1,2,3,4,7,17,19,20,21,25,29,31,11,13]
    total_channels = np.arange(40)
    del_channels = list(set(useful_channels) ^ set(total_channels))
    train_data_load = np.delete(eegvals, del_channels, 1)
    train_data_load = np.swapaxes(train_data_load, 1, 2)
    train_data = np.concatenate((train_data, train_data_load), 0)
    train_labels = np.concatenate((train_labels, eeglabels),0)

# %% prepare labels
    
path_to_labels = 'C:/Users/Dell/Desktop/labels_new.csv'
csv_file = open(path_to_labels, 'r')
new_labels = list()
for data in csv_file:
    print(data)
    data = data.split(',')
    if data[len(data) - 1] == 'hi\n':
        new_labels.append(1)
    elif data[len(data) - 1] == 'lo\n':
        new_labels.append(0)
    else:
        new_labels.append(2)
new_labels = new_labels[1:]
if 2 in new_labels:
    print('Issues detected')
labels = list()
for i in range(26):
    labels.append(new_labels)
labels = np.resize(np.array(labels),[1040,1])


# %% prepare test-validation ratio

train_data = train_data[1:]
train_labels = train_labels[1:]
train_data, train_labels = shuffle(train_data, train_labels)
train_data = train_data / float(np.max(train_data))

# %% try and get the nine elements to fabricate the dataset, in a way

train_data_new = list()
for i in range(1040):
    data_element = list()
    for j in range(14):
        mean = statistics.mean(train_data[i,:,j])
        median = statistics.median(train_data[i,:,j])
        maximum = np.max(train_data[i,:,j])
        minimum = np.min(train_data[i,:,j])
        std = statistics.stdev(train_data[i,:,j])
        variance = statistics.variance(train_data[i,:,j])
        data_element.append([mean, median, maximum, minimum, std, variance])
    train_data_new.append(data_element)

# %% model

model = models.Sequential()
model.add(layers.Conv1D(32, 3, strides=2, activation='relu', input_shape=(8064,14)))
model.add(layers.Conv1D(32, 3, strides=2, activation='relu'))
model.add(layers.MaxPooling1D(2))
model.add(layers.Conv1D(64, 3, activation='relu'))
model.add(layers.Conv1D(64, 3, activation='relu'))
model.add(layers.MaxPooling1D(2))
model.add(layers.Conv1D(96, 3, activation='relu'))
model.add(layers.Conv1D(96, 3, activation='relu'))
model.add(layers.MaxPooling1D(2))
model.add(layers.Conv1D(128, 3, activation='relu'))
model.add(layers.Conv1D(128, 3, activation='relu'))
model.add(layers.MaxPooling1D(2))
model.add(layers.Conv1D(256, 3, activation='relu'))
model.add(layers.Conv1D(256, 3, activation='relu'))
model.add(layers.MaxPooling1D(2))
model.add(layers.Conv1D(312, 3, activation='relu'))
model.add(layers.Conv1D(312, 3, activation='relu'))
model.add(layers.MaxPooling1D(2))
model.add(layers.Conv1D(512, 3, activation='relu'))
model.add(layers.Conv1D(512, 3, activation='relu'))
#model.add(layers.MaxPooling1D(2))
#model.add(layers.Conv1D(864, 3, activation='relu'))
#model.add(layers.Conv1D(864, 3, activation='relu'))
model.add(layers.MaxPooling1D(2))
model.add(layers.Conv1D(1024, 3, activation='relu'))
model.add(layers.Conv1D(1024, 3, activation='relu'))
model.add(layers.MaxPooling1D(2))
#model.add(layers.Conv1D(2048, 3, activation='relu'))
#model.add(layers.Conv1D(2048, 3, activation='relu'))
#model.add(layers.MaxPooling1D(2))
model.add(layers.Flatten())
#model.add(layers.Dense(5024, activation='relu'))
#model.add(layers.Dense(1000, activation='relu'))
model.add(layers.Dense(64, activation='relu'))
model.add(layers.Dense(64, activation='relu'))
#model.add(layers.Dense(125, activation='relu'))
#model.add(layers.Dense(50, activation='relu'))
#model.add(layers.Dense(10, activation='relu'))

model.add(layers.Dense(4))

# %% model 2

model = models.Sequential()
model.add(layers.Conv1D(32, 3, strides=2, activation='relu', input_shape=(8064,14)))
model.add(layers.Conv1D(32, 3, strides=2, activation='relu'))
model.add(layers.MaxPooling1D(2))
model.add(layers.BatchNormalization())
model.add(layers.Conv1D(64, 3, strides=2, activation='relu'))
model.add(layers.Conv1D(64, 3, strides=2, activation='relu'))
model.add(layers.MaxPooling1D(2))
model.add(layers.BatchNormalization())
model.add(layers.Conv1D(96, 3, strides=2, activation='relu'))
model.add(layers.Conv1D(96, 3, strides=2, activation='relu'))
model.add(layers.MaxPooling1D(2))
model.add(layers.BatchNormalization())
model.add(layers.Conv1D(128, 3, strides=2, activation='relu'))
model.add(layers.Conv1D(128, 3, strides=2, activation='relu'))
model.add(layers.MaxPooling1D(2))
model.add(layers.BatchNormalization())
#model.add(layers.Conv1D(256, 3, activation='relu'))
#model.add(layers.Conv1D(256, 3, activation='relu'))
#model.add(layers.MaxPooling1D(2))
#model.add(layers.Conv1D(312, 3, activation='relu'))
#model.add(layers.Conv1D(312, 3, activation='relu'))
#model.add(layers.MaxPooling1D(2))
#model.add(layers.Conv1D(512, 3, activation='relu'))
#model.add(layers.Conv1D(512, 3, activation='relu'))
#model.add(layers.MaxPooling1D(2))
#model.add(layers.Conv1D(864, 3, activation='relu'))
#model.add(layers.Conv1D(864, 3, activation='relu'))
#model.add(layers.MaxPooling1D(2))
#model.add(layers.Conv1D(1024, 3, activation='relu'))
#model.add(layers.Conv1D(1024, 3, activation='relu'))
#model.add(layers.MaxPooling1D(2))
#model.add(layers.Conv1D(2048, 3, activation='relu'))
#model.add(layers.Conv1D(2048, 3, activation='relu'))
#model.add(layers.MaxPooling1D(2))
model.add(layers.Flatten())
#model.add(layers.Dense(5024, activation='relu'))
#model.add(layers.Dense(1000, activation='relu'))
model.add(layers.Dense(64, activation='relu'))
model.add(layers.Dropout(0.3))
model.add(layers.Dense(64, activation='relu'))
model.add(layers.Dropout(0.3))
#model.add(layers.Dense(125, activation='relu'))
#model.add(layers.Dense(50, activation='relu'))
#model.add(layers.Dense(10, activation='relu'))

model.add(layers.Dense(2, activation='softmax'))

# %% new statistics model

model = models.Sequential()
model.add(layers.Flatten(input_shape=(14,6)))
model.add(layers.Dense(64, activation='relu'))
model.add(layers.Dense(32, activation='relu'))
model.add(layers.Dense(4))

# %% statistical model with conv1D

model = models.Sequential()
model.add(layers.Conv1D(32, 3, activation='relu', input_shape=(14,6)))
model.add(layers.Conv1D(32, 3, activation='relu'))
model.add(layers.MaxPooling1D(2))
model.add(layers.Conv1D(64, 3, activation='relu'))
model.add(layers.Conv1D(64, 3, activation='relu'))
model.add(layers.Flatten())
model.add(layers.Dense(64, activation='relu'))
model.add(layers.Dense(4))

# %% testing to see if a false local minima reached

model = models.Sequential()
model.add(layers.Flatten(input_shape=(14,6)))
model.add(layers.Dense(84, activation="relu"))
model.add(layers.Dense(4))

# %% train

model.compile(metrics=['mae', 'mse'], loss='mse', optimizer='adam')#optimizer=tf.keras.optimizers.SGD(lr=1e-3))
log_dir = "D:/Mini II/TensorBoard_Data/"
tensorboard_callbacks = tf.keras.callbacks.TensorBoard(log_dir=log_dir, histogram_freq=1)
model.fit(train_data, train_labels, epochs=50, validation_split = 0.1, callbacks=[TrainValTensorBoard(write_graph=False)])

# %% train_classifier

model.compile(metrics=['accuracy'], optimizer='adam', loss='sparse_categorical_crossentropy')
log_dir = "D:/Mini II/TensorBoard_Data/"
tensorboard_callbacks = tf.keras.callbacks.TensorBoard(log_dir=log_dir, histogram_freq=1)
model.fit(train_data, labels, epochs=30, callbacks=[TrainValTensorBoard(write_graph=False)], validation_split = 0.1)

# %% val + train graph
import os
from tensorflow.keras.callbacks import TensorBoard

class TrainValTensorBoard(TensorBoard):
    def __init__(self, log_dir='.\logs', **kwargs):
        # Make the original `TensorBoard` log to a subdirectory 'training'
        training_log_dir = os.path.join(log_dir, 'training')
        super(TrainValTensorBoard, self).__init__(training_log_dir, **kwargs)

        # Log the validation metrics to a separate subdirectory
        self.val_log_dir = os.path.join(log_dir, 'validation')

    def set_model(self, model):
        # Setup writer for validation metrics
        self.val_writer = tf.summary.FileWriter(self.val_log_dir)
        super(TrainValTensorBoard, self).set_model(model)

    def on_epoch_end(self, epoch, logs=None):
        # Pop the validation logs and handle them separately with
        # `self.val_writer`. Also rename the keys so that they can
        # be plotted on the same figure with the training metrics
        logs = logs or {}
        val_logs = {k.replace('val_', ''): v for k, v in logs.items() if k.startswith('val_')}
        for name, value in val_logs.items():
            summary = tf.Summary()
            summary_value = summary.value.add()
            summary_value.simple_value = value.item()
            summary_value.tag = name
            self.val_writer.add_summary(summary, epoch)
        self.val_writer.flush()

        # Pass the remaining logs to `TensorBoard.on_epoch_end`
        logs = {k: v for k, v in logs.items() if not k.startswith('val_')}
        super(TrainValTensorBoard, self).on_epoch_end(epoch, logs)

    def on_train_end(self, logs=None):
        super(TrainValTensorBoard, self).on_train_end(logs)
        self.val_writer.close()

# %% test

test_data = np.zeros((1,8064,14))
test_labels = np.zeros((1,4))
for i in range(26,31):
    module = open('D:\Mini_II\DATASETS\DEAP\data_preprocessed_python\data_preprocessed_python\s{0:02d}.dat'.format(i+1), 'rb')
    dataset = pickle.load(module, encoding='bytes')
    eegvals = np.array(dataset[b'data'])
    eeglabels = np.array(dataset[b'labels'])
    useful_channels = [1,2,3,4,7,17,19,20,21,25,29,31,11,13]
    total_channels = np.arange(40)
    del_channels = list(set(useful_channels) ^ set(total_channels))
    test_data_load = np.delete(eegvals, del_channels, 1)
    test_data_load = np.swapaxes(test_data_load, 1, 2)
    test_data = np.concatenate((test_data, test_data_load), 0)
    test_labels = np.concatenate((test_labels, eeglabels),0)

# %% prep
    
test_data_new = list()
for i in range(200):
    data_element = list()
    for j in range(14):
        mean = statistics.mean(test_data[i,:,j])
        median = statistics.median(test_data[i,:,j])
        maximum = np.max(test_data[i,:,j])
        minimum = np.min(test_data[i,:,j])
        std = statistics.stdev(test_data[i,:,j])
        variance = statistics.variance(test_data[i,:,j])
        data_element.append([mean, median, maximum, minimum, std, variance])
    test_data_new.append(data_element)

# %% test set test
    
loss = model.evaluate(test_data, test_labels)
print(loss)