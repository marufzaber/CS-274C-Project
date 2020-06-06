import imageio
import os
import numpy as np

from project.fma import utils
#from project.tools import graph_generator

import multiprocessing.sharedctypes as sharedctypes
import ctypes
import tensorflow.keras as keras
from tensorflow.keras.layers import Activation, Dense, Conv1D, Conv2D, Conv3D, AveragePooling1D, MaxPooling1D, MaxPooling2D, MaxPooling3D, Flatten, Reshape, BatchNormalization
from tensorflow.keras.layers import LSTM, TimeDistributed, Dropout, GRU, GaussianNoise

from tensorflow.keras import regularizers

from .base import AUDIO_DIR, AUDIO_META_DIR, preprocess
from ..tools import graph_generator

class MelSpectrogramLoader(utils.RawAudioLoader):
    def __init__(self, *args, **kwargs):
        super(MelSpectrogramLoader, self).__init__(*args, **kwargs)
        #self.shape = [480, 640, 4]
        self.shape = [480, 640]

    def _load(self, filepath):
        img3d = imageio.imread(filepath, format="PNG-PIL")
        return np.average(img3d, axis=2).reshape([480, 640])
        return img3d
        # layer0 = img3d[:, :, 0]
        # layer1 = img3d[:, :, 1]
        # layer2 = img3d[:, :, 2]
        # layer3 = img3d[:, :, 3]
        # # img2d = np.empty(self.shape, dtype=layer0.dtype)
        # img2d[0::4, :] = layer0
        # img2d[1::4, :] = layer1
        # img2d[2::4, :] = layer2
        # img2d[3::4, :] = layer3
        # import pdb; pdb.set_trace()
        # return img2d

def train(num_epochs, batch_size, learning_rate, output_dir):
    train, val, test, labels_onehot = preprocess()
    print(train)

    # Some Parameters commented out for Windows
    # Keras parameters.
    #NB_WORKER = len(os.sched_getaffinity(0))  # number of usables CPUs
    params = { 'max_queue_size': 10} # 'workers': NB_WORKER,

    # Parameters for Model
    LSTM_COUNT = 128
    DENSE_COUNT = 64
    NUM_CLASSES = 8
    L2_regularization = 0.001


    #print('Dimensionality: {}'.format(loader.shape))
    #SampleLoader = utils.build_sample_loader(AUDIO_DIR, labels_onehot, utils.FfmpegLoader())
    SampleLoader = utils.build_sample_loader(AUDIO_DIR, labels_onehot, MelSpectrogramLoader(), extension="png")

    #loader = utils.FfmpegLoader(sampling_rate=2000)
    #print('Dimensionality: {}'.format(loader.shape))
    #SampleLoader = utils.build_sample_loader(AUDIO_DIR, labels_onehot, utils.FfmpegLoader())


    keras.backend.clear_session()

    model = keras.Sequential()
    
    # LSTM layer
    #model.add(GaussianNoise)
    model.add(LSTM(LSTM_COUNT, return_sequences=True)) # can substitute SimpleRNN or GRU for LSTM layer
    model.add(LSTM(LSTM_COUNT, return_sequences=True))
    model.add(Dropout(0.4))

    # Dense layer
    model.add((Dense(DENSE_COUNT, activation='relu'))) # kernel_regularizer=regularizers.l2(L2_regularization), TimeDistributed
    model.add((Dense(DENSE_COUNT/2, activation='relu'))) # kernel_regularizer=regularizers.l2(L2_regularization), TimeDistributed
    model.add((Dense(DENSE_COUNT/4, activation='relu'))) # kernel_regularizer=regularizers.l2(L2_regularization), TimeDistributed
    #model.add(Dropout(0.4))
    model.add(Flatten())

    # Softmax Layer
    model.add(Dense(NUM_CLASSES, activation='softmax'))

    optimizer = keras.optimizers.Adam(lr=learning_rate)#, momentum=0.9, nesterov=True)
    model.compile(optimizer, loss='categorical_crossentropy', metrics=['accuracy'])
    
    #history = model.fit_generator(SampleLoader(train, batch_size=batch_size), train.size/batch_size, epochs=num_epochs, **params)

    total_epochs = 0
    
    while (total_epochs < num_epochs):
        training_history = model.fit_generator(SampleLoader(train, batch_size=batch_size), train.size/batch_size, epochs=4, **params).history
        val_loss, val_acc = model.evaluate_generator(SampleLoader(val, batch_size=batch_size), val.size, **params)

        for ep in range(4):
            graph_generator.store_in_csv(total_epochs + ep + 1, batch_size, learning_rate, training_history['accuracy'][ep], training_history['loss'][ep], os.path.join(output_dir, "lstm_training_data.csv"))
        graph_generator.store_in_csv(total_epochs + 4, batch_size, learning_rate, val_acc, val_loss, os.path.join(output_dir, "lstm_validation_data.csv"))
        total_epochs += 4
    graph_generator.generate_from_csv(os.path.join(output_dir, "lstm_training_data.csv"), os.path.join(output_dir, "lstm_training_plot.eps"), labels_prefix='Training', color='r')
    graph_generator.generate_from_csv(os.path.join(output_dir, "lstm_validation_data.csv"), os.path.join(output_dir, "lstm_combined_plot.eps"), labels_prefix='Validation', color='b', epoch_step=4)
    print(model.summary(to_file=os.path.join(output_dir, 'lstm_model_summary.txt')))

    #model.save(os.path.join(job_dir, 'model-export'), save_format='tf')
    #acc = model.evaluate_generator(SampleLoader(val, batch_size=100), val.size, **params)
    acc = model.evaluate_generator(SampleLoader(val, batch_size=100), val.size, **params)
    print( "**** VAL ACC ****")
    print(acc)
    print(model.summary())
    return model