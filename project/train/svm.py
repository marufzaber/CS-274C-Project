import os
import numpy as np

from project.fma import utils
import multiprocessing.sharedctypes as sharedctypes
import ctypes
import tensorflow.keras as keras
from tensorflow.keras.layers import Activation, Dense, Conv1D, Conv2D, MaxPooling1D, Flatten, Reshape

from .base import AUDIO_DIR, AUDIO_META_DIR, preprocess

def train(num_epochs, batch_size, learning_rate, job_dir):
    train, val, test, labels_onehot = preprocess()

    NB_WORKER = len(os.sched_getaffinity(0))  # number of usables CPUs
    params = {'workers': NB_WORKER, 'max_queue_size': 10}

    loader = utils.FfmpegLoader(sampling_rate=2000)
    SampleLoader = utils.build_sample_loader(AUDIO_DIR, labels_onehot, utils.FfmpegLoader())

    keras.backend.clear_session()

    model = keras.Sequential(
        [Dense(100, input_shape=loader.shape, activation="relu"),
        Dense(100, W_regularizer=l2(0.01)),
        Dense(labels_onehot.shape[1], activation="linear")]
               
    model.compile(loss='squared_hinge', optimizer='adadelta', metrics=['accuracy'])

    model.fit_generator(SampleLoader(train, batch_size=batch_size), train.size/batch_size, epochs=num_epochs, **params)

    model.save(os.path.join(job_dir, 'model-export'), save_format='tf')