import ast
import os

import numpy as np
import pandas as pd
import keras
from keras.layers import Activation, Dense, Conv1D, Conv2D, MaxPooling1D, Flatten, Reshape
from sklearn.preprocessing import MultiLabelBinarizer, LabelEncoder, LabelBinarizer, StandardScaler
import multiprocessing.sharedctypes as sharedctypes
import ctypes


from project.fma import utils

AUDIO_DIR = os.environ.get('AUDIO_DIR')
AUDIO_META_DIR = os.environ.get('AUDIO_META_DIR')


def train(num_epochs, batch_size, learning_rate, job_dir):
                                        #
    tracks = utils.load(os.path.join(AUDIO_META_DIR, 'tracks.csv'))
    small = tracks['set', 'subset'] <= 'small'
    have_genres = tracks['track', 'genre_top'] != 'MISSING'
    #longer = tracks['track', 'duration'] >= 100
    tracks = tracks[small & have_genres]
    features = utils.load(os.path.join(AUDIO_META_DIR, 'features.csv'))
    #small = features['set', 'subset'] <= 'small'
    features = features[small & have_genres]
    echonest = utils.load(os.path.join(AUDIO_META_DIR, 'echonest.csv'))
    #small = echonest['set', 'subset'] <= 'small'
    echonest = echonest[small & have_genres]

    np.testing.assert_array_equal(features.index, tracks.index)
    assert echonest.index.isin(tracks.index).all()

    tracks.shape, features.shape, echonest.shape

    labels_onehot = LabelBinarizer().fit_transform(tracks['track', 'genre_top'])
    labels_onehot = pd.DataFrame(labels_onehot, index=tracks.index)


    train = tracks.index[tracks['set', 'split'] == 'training']
    val = tracks.index[tracks['set', 'split'] == 'validation']
    test = tracks.index[tracks['set', 'split'] == 'test']

    print('{} training examples, {} validation examples, {} testing examples'.format(*map(len, [train, val, test])))

    genres = list(LabelEncoder().fit(tracks['track', 'genre_top']).classes_)
    #genres = list(tracks['track', 'genre_top'].unique())
    print('Top genres ({}): {}'.format(len(genres), genres))
    #genres = list(MultiLabelBinarizer().fit(tracks['track', 'genres_all']).classes_)
    #print('All genres ({}): {}'.format(len(genres), genres))

    # Just be sure that everything is fine. Multiprocessing is tricky to debug.
    utils.FfmpegLoader().load(utils.get_audio_path(AUDIO_DIR, 2))
    SampleLoader = utils.build_sample_loader(AUDIO_DIR, labels_onehot, utils.FfmpegLoader())


    #
    # Keras parameters.
    NB_WORKER = len(os.sched_getaffinity(0))  # number of usables CPUs
    params = {'pickle_safe': True, 'nb_worker': NB_WORKER, 'max_q_size': 10}


    loader = utils.FfmpegLoader(sampling_rate=2000)
    #SampleLoader = utils.build_sample_loader(AUDIO_DIR, labels_onehot, loader)
    print('Dimensionality: {}'.format(loader.shape))
    SampleLoader = utils.build_sample_loader(AUDIO_DIR, labels_onehot, utils.FfmpegLoader())
    sl = SampleLoader(train, batch_size=100)


    floader = utils.FfmpegLoader(sampling_rate=2000)
    X = np.empty((10000, *floader.shape))
    data = sharedctypes.RawArray(ctypes.c_int, train)

    bad = tracks.index.isin([99134, 108925, 133297])
    training = tracks['set', 'split'] == 'training'
    train = tracks[training & ~bad].head(100).index


    keras.backend.clear_session()

    model = keras.models.Sequential()
    model.add(Dense(output_dim=100, input_shape=loader.shape))
    model.add(Activation("relu"))
    model.add(Dense(output_dim=100))
    model.add(Activation("relu"))
    model.add(Dense(output_dim=labels_onehot.shape[1]))
    model.add(Activation("softmax"))
    #
    optimizer = keras.optimizers.SGD(lr=learning_rate)#, momentum=0.9, nesterov=True)
    model.compile(optimizer, loss='categorical_crossentropy', metrics=['accuracy'])
    #
    model.fit_generator(SampleLoader(train, batch_size=batch_size), train.size, nb_epoch=num_epochs, **params)

    model.save(os.path.join(job_dir, 'model-export'), save_format='tf')
    #loss = model.evaluate_generator(SampleLoader(val, batch_size=64), val.size, **params)
    #loss = model.evaluate_generator(SampleLoader(test, batch_size=64), test.size, **params)
    #Y = model.predict_generator(SampleLoader(test, batch_size=64), test.size, **params);
    #
    # loss

    #
