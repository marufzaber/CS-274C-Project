import os
import numpy as np

from project.fma import utils
from project.tools import graph_generator

import multiprocessing.sharedctypes as sharedctypes
import ctypes
import tensorflow.keras as keras
from tensorflow.keras.layers import Dropout, Activation, Dense, Reshape, Flatten
from keras.regularizers import l2
import psutil
from tensorflow.keras.layers.experimental import RandomFourierFeatures

from keras import regularizers

from .base import AUDIO_DIR, AUDIO_META_DIR, preprocess


def train(num_epochs, batch_size, learning_rate, job_dir):
    train, val, test, labels_onehot = preprocess()

    NB_WORKER = psutil.cpu_count()  # number of usables CPUs
    params = {'workers': NB_WORKER, 'max_queue_size': 10}

    loader = utils.FfmpegLoader(sampling_rate=2000)
    SampleLoader = utils.build_sample_loader(AUDIO_DIR, labels_onehot, utils.FfmpegLoader())

    keras.backend.clear_session()

    #model = keras.Sequential()

    model = keras.Sequential([RandomFourierFeatures(output_dim=4096, scale=10.0, kernel_initializer="gaussian")])

    model.add(Dense(1024, input_shape=loader.shape, activation="relu"))   
    model.add(Dropout(0.5))
    model.add(Dense(labels_onehot.shape[1], kernel_regularizer=regularizers.l2(0.01), activation="linear"))

    model.compile(optimizer=keras.optimizers.Adam(learning_rate=1e-3), loss='hinge', metrics=['accuracy'])

    history = model.fit_generator(SampleLoader(train, batch_size=batch_size), train.size/batch_size, epochs=num_epochs, **params)

    model.save(os.path.join(job_dir, 'model-export'), save_format='tf')

    acc = model.evaluate_generator(SampleLoader(val, batch_size=100), val.size, **params)
    
    print( "**** VAL ACC ****")
    print(acc)

    history_dict = history.history
