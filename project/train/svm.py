import os
import numpy as np

from project.fma import utils
from project.tools import graph_generator

import multiprocessing.sharedctypes as sharedctypes
import ctypes
import tensorflow.keras as keras
from tensorflow.keras.layers import Dropout, Activation, Dense, Reshape
from keras.regularizers import l2

from .base import AUDIO_DIR, AUDIO_META_DIR, preprocess


def train(num_epochs, batch_size, learning_rate, job_dir):
    train, val, test, labels_onehot = preprocess()

    NB_WORKER = len(os.sched_getaffinity(0))  # number of usables CPUs
    params = {'workers': NB_WORKER, 'max_queue_size': 10}

    loader = utils.FfmpegLoader(sampling_rate=2000)
    SampleLoader = utils.build_sample_loader(AUDIO_DIR, labels_onehot, utils.FfmpegLoader())

    keras.backend.clear_session()

    model = keras.Sequential()

    model.add(Dense(100, input_shape=loader.shape, activation="relu"))
    model.add(Dropout(0.5))
    model.add(Dense(100, kernel_regularizer=l2(0.01), activation="relu"))
    model.add(Dropout(0.5))
    model.add(Dense(labels_onehot.shape[1], activation="sigmoid"))

    model.compile(optimizer='adadelta', loss='squared_hinge', metrics=['accuracy'])

    history = model.fit_generator(SampleLoader(train, batch_size=batch_size), train.size/batch_size, epochs=num_epochs, **params)

    model.save(os.path.join(job_dir, 'model-export'), save_format='tf')

    history_dict = history.history

    print(history_dict.keys())

