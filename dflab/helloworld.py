import os

import numpy as np
import pandas as pd
import keras
from keras.layers import Activation, Dense, Conv1D, Conv2D, MaxPooling1D, Flatten, Reshape
from sklearn.preprocessing import MultiLabelBinarizer, LabelEncoder, LabelBinarizer, StandardScaler

from fma import utils

AUDIO_DIR = os.environ.get('AUDIO_DIR')
AUDIO_META_DIR = os.environ.get('AUDIO_META_DIR')

tracks = utils.load(os.path.join(AUDIO_META_DIR, 'tracks.csv'))
features = utils.load(os.path.join(AUDIO_META_DIR, 'features.csv'))
echonest = utils.load(os.path.join(AUDIO_META_DIR, 'echonest.csv'))

np.testing.assert_array_equal(features.index, tracks.index)
assert echonest.index.isin(tracks.index).all()

tracks.shape, features.shape, echonest.shape

labels_onehot = LabelBinarizer().fit_transform(tracks['track', 'genre_top'])
labels_onehot = pd.DataFrame(labels_onehot, index=tracks.index)


# Just be sure that everything is fine. Multiprocessing is tricky to debug.
utils.FfmpegLoader().load(utils.get_audio_path(AUDIO_DIR, 2))
SampleLoader = utils.build_sample_loader(AUDIO_DIR, labels_onehot, utils.FfmpegLoader())
SampleLoader(train, batch_size=2).__next__()[0].shape


# Keras parameters.
NB_WORKER = len(os.sched_getaffinity(0))  # number of usables CPUs
params = {'pickle_safe': True, 'nb_worker': NB_WORKER, 'max_q_size': 10}


loader = utils.FfmpegLoader(sampling_rate=2000)
SampleLoader = utils.build_sample_loader(AUDIO_DIR, labels_onehot, loader)
print('Dimensionality: {}'.format(loader.shape))

keras.backend.clear_session()

model = keras.models.Sequential()
model.add(Dense(output_dim=1000, input_shape=loader.shape))
model.add(Activation("relu"))
model.add(Dense(output_dim=100))
model.add(Activation("relu"))
model.add(Dense(output_dim=labels_onehot.shape[1]))
model.add(Activation("softmax"))

optimizer = keras.optimizers.SGD(lr=0.1, momentum=0.9, nesterov=True)
model.compile(optimizer, loss='categorical_crossentropy', metrics=['accuracy'])

model.fit_generator(SampleLoader(train, batch_size=64), train.size, nb_epoch=2, **params)
loss = model.evaluate_generator(SampleLoader(val, batch_size=64), val.size, **params)
loss = model.evaluate_generator(SampleLoader(test, batch_size=64), test.size, **params)
#Y = model.predict_generator(SampleLoader(test, batch_size=64), test.size, **params);

loss


# classifiers = {
#     'LR': LogisticRegression(),
#     'kNN': KNeighborsClassifier(n_neighbors=200),
#     'SVCrbf': SVC(kernel='rbf'),
#     'SVCpoly1': SVC(kernel='poly', degree=1),
#     'linSVC1': SVC(kernel="linear"),
#     'linSVC2': LinearSVC(),
#     #GaussianProcessClassifier(1.0 * RBF(1.0), warm_start=True),
#     'DT': DecisionTreeClassifier(max_depth=5),
#     'RF': RandomForestClassifier(max_depth=5, n_estimators=10, max_features=1),
#     'AdaBoost': AdaBoostClassifier(n_estimators=10),
#     'MLP1': MLPClassifier(hidden_layer_sizes=(100,), max_iter=2000),
#     'MLP2': MLPClassifier(hidden_layer_sizes=(200, 50), max_iter=2000),
#     'NB': GaussianNB(),
#     'QDA': QuadraticDiscriminantAnalysis(),
# }
#
#
# def pre_process(tracks, features, columns, multi_label=False, verbose=False):
#     if not multi_label:
#         # Assign an integer value to each genre.
#         enc = LabelEncoder()
#         labels = tracks['track', 'genre_top']
#         # y = enc.fit_transform(tracks['track', 'genre_top'])
#     else:
#         # Create an indicator matrix.
#         enc = MultiLabelBinarizer()
#         labels = tracks['track', 'genres_all']
#         # labels = tracks['track', 'genres']
#
#     # Split in training, validation and testing sets.
#     y_train = enc.fit_transform(labels[train])
#     y_val = enc.transform(labels[val])
#     y_test = enc.transform(labels[test])
#     X_train = features.loc[train, columns].as_matrix()
#     X_val = features.loc[val, columns].as_matrix()
#     X_test = features.loc[test, columns].as_matrix()
#
#     X_train, y_train = shuffle(X_train, y_train, random_state=42)
#
#     # Standardize features by removing the mean and scaling to unit variance.
#     scaler = StandardScaler(copy=False)
#     scaler.fit_transform(X_train)
#     scaler.transform(X_val)
#     scaler.transform(X_test)
#
#     return y_train, y_val, y_test, X_train, X_val, X_test
#
#
# def test_classifiers_features(classifiers, feature_sets, multi_label=False):
#     columns = list(classifiers.keys()).insert(0, 'dim')
#     scores = pd.DataFrame(columns=columns, index=feature_sets.keys())
#     times = pd.DataFrame(columns=classifiers.keys(), index=feature_sets.keys())
#     for fset_name, fset in tqdm_notebook(feature_sets.items(), desc='features'):
#         y_train, y_val, y_test, X_train, X_val, X_test = pre_process(tracks, features_all, fset, multi_label)
#         scores.loc[fset_name, 'dim'] = X_train.shape[1]
#         for clf_name, clf in classifiers.items():  # tqdm_notebook(classifiers.items(), desc='classifiers', leave=False):
#             t = time.process_time()
#             clf.fit(X_train, y_train)
#             score = clf.score(X_test, y_test)
#             scores.loc[fset_name, clf_name] = score
#             times.loc[fset_name, clf_name] = time.process_time() - t
#     return scores, times

