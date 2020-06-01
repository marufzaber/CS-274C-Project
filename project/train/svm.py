import os
import numpy as np

from project.fma import utils
from project.tools import graph_generator

import tensorflow.keras as keras
from tensorflow.keras.layers import Dropout, Activation, Dense, Reshape, Flatten
from tensorflow.keras import initializers
from keras.regularizers import l2
import psutil
from tensorflow.keras.layers.experimental import RandomFourierFeatures
from tensorflow.keras import initializers
from sklearn.utils import shuffle

from keras import regularizers


from .base import AUDIO_DIR, AUDIO_META_DIR, preprocess


def train(num_epochs, batch_size, learning_rate, job_dir):
    train, val, test, labels_onehot = preprocess()

    train = shuffle(train)

    NB_WORKER = psutil.cpu_count()  # number of usables CPUs
    params = {'workers': NB_WORKER, 'max_queue_size': 10}

    loader = utils.FfmpegLoader(sampling_rate=2000)
    SampleLoader = utils.build_sample_loader(AUDIO_DIR, labels_onehot, utils.FfmpegLoader())

    keras.backend.clear_session()

    #model = keras.Sequential()

    model = keras.Sequential([RandomFourierFeatures(
        output_dim=2048, 
        scale=10.0, 
        kernel_initializer="laplacian")]
    )

    model.add(Dense(1000, input_shape=loader.shape, activation="relu"))   
    model.add(Dropout(0.5))

    model.add(Dense(
        labels_onehot.shape[1], 
        kernel_initializer=initializers.RandomNormal(stddev=0.01),
        bias_initializer=initializers.Zeros(), 
        kernel_regularizer=regularizers.l2(0.01), 
        activity_regularizer=regularizers.l2(1e-5),
        bias_regularizer=regularizers.l2(1e-4),
        activation="softmax"))

    model.compile(
        optimizer=keras.optimizers.Adam(learning_rate=1e-3), 
        loss='categorical_hinge', 
        metrics=['accuracy']
    )

    history = model.fit_generator(
        SampleLoader(train, batch_size=batch_size), 
        train.size/batch_size, 
        epochs=num_epochs, 
        **params
    )

    model.save(os.path.join(job_dir, 'model-export'), save_format='tf')

    val_acc = model.evaluate_generator (
        SampleLoader(val, batch_size=100), 
        val.size, 
        **params
    ) 

    print( "**** VAL ACC ****")
    print(val_acc)

    val_acc = [0,1]
    
    graph_generator.store_in_csv(num_epochs, batch_size, learning_rate, val_acc[1], val_acc[0], "/Users/demigorgan/Desktop/CS-274C-Project/validation_data.csv")

    test_acc = model.evaluate_generator(
        SampleLoader(test, batch_size=100), 
        test.size, 
        **params)
    print( "**** TEST ACC ****")
    print(test_acc)

    graph_generator.store_in_csv(num_epochs, batch_size, learning_rate, test_acc[1], test_acc[0], "/Users/demigorgan/Desktop/CS-274C-Project/test_data.csv")


    history_dict = history.history

