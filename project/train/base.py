import os

import numpy as np
import pandas as pd
from sklearn.preprocessing import MultiLabelBinarizer, LabelEncoder, LabelBinarizer, StandardScaler

from tensorflow.io import gfile

from project.fma import utils

JOB_DIR = os.environ.get('JOB_DIR', 'keras-job-dir')
#FULL_JOB_DIR = os.environ.get('FULL_JOB_DIR', 'gs://helloworld-ucics274c-df-mybucket/keras-job-dir')
FULL_JOB_DIR = 'gs://helloworld-ucics274c-df-mybucket/keras-job-dir'
# print("READING FILE")
# print(gfile.listdir('.'))
# print(gfile.listdir('gs://helloworld-ucics274c-df-mybucket'))
# # print(gfile.listdir(JOB_DIR))
# print(gfile.listdir(FULL_JOB_DIR))
# # with utils.open_file('fma_metadata/fma_metadata.tracks.csv') as f:
# #     print(f.readline())
# with utils.open_file(f'{FULL_JOB_DIR}/fma_metadata/fma_metadata.tracks.csv') as f:
#     print(f.readline())
AUDIO_DIR = f'fma_small'
AUDIO_META_DIR = f'fma_metadata/fma_metadata'
#AUDIO_DIR = os.environ.get('AUDIO_DIR', f'{FULL_JOB_DIR}/fma_small')
#AUDIO_META_DIR = os.environ.get('AUDIO_META_DIR', f'{FULL_JOB_DIR}/fma_metadata/fma_metadata')
# with utils.open_file(f'{FULL_JOB_DIR}/fma_metadata/fma_metadata/tracks.csv') as f:
#     print(f.readline())
# filepath = f'{FULL_JOB_DIR}/fma_small/000/000002.png'
# gfile.copy(filepath, 'tmp_audio.png', overwrite=True)
# with utils.open_file(filepath, 'rb') as f:
#     with open('tmp_audio.png', 'wb') as of:
#         of.write(f.read())
# import librosa
# x, sr = librosa.load('tmp_audio.png', sr=None)
# print(x)

def preprocess(job_dir):
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
    train = tracks[training & ~bad].index


    return train, val, test, labels_onehot