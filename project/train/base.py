import os

import numpy as np
import pandas as pd
from sklearn.preprocessing import MultiLabelBinarizer, LabelEncoder, LabelBinarizer, StandardScaler



from project.fma import utils

JOB_DIR = os.environ.get('JOB_DIR', 'gs://helloworld-ucics274c-df-mybucket/keras-job-dir')
AUDIO_DIR = os.environ.get('AUDIO_DIR', f'{JOB_DIR}/fma_small/fma_small')
AUDIO_META_DIR = os.environ.get('AUDIO_META_DIR', f'{JOB_DIR}/fma_metadata/fma_metadata')


def preprocess():
                                       #
    tracks = utils.load(os.path.join(AUDIO_META_DIR, 'tracks.csv'))
    small = tracks['set', 'subset'] <= 'small'
    have_genres = tracks['track', 'genre_top'] != 'MISSING'
    tracks = tracks[small & have_genres]
    features = utils.load(os.path.join(AUDIO_META_DIR, 'features.csv'))
    features = features[small & have_genres]
    echonest = utils.load(os.path.join(AUDIO_META_DIR, 'echonest.csv'))
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

    bad = tracks.index.isin([2, 56, 99134, 108925, 133297])
    training = tracks['set', 'split'] == 'training'
    train = tracks[training & ~bad].head(300).index


    return train, val, test, labels_onehot