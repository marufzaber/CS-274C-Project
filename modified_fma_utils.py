import dotenv
import ffmpeg
import pydot
import requests
import numpy as np
import pandas as pd
import ctypes
import shutil
import multiprocessing
import multiprocessing.sharedctypes as sharedctypes
import os.path
import ast


# Number of samples per 30s audio clip.
# TODO: fix dataset to be constant.
NB_AUDIO_SAMPLES = 1000
SAMPLING_RATE = 2000

# Load the environment from the .env file.
dotenv.load_dotenv(dotenv.find_dotenv())


class FreeMusicArchive:

    BASE_URL = 'https://freemusicarchive.org/api/get/'

    def __init__(self, api_key):
        self.api_key = api_key

    def get_recent_tracks(self):
        URL = 'https://freemusicarchive.org/recent.json'
        r = requests.get(URL)
        r.raise_for_status()
        tracks = []
        artists = []
        date_created = []
        for track in r.json()['aTracks']:
            tracks.append(track['track_id'])
            artists.append(track['artist_name'])
            date_created.append(track['track_date_created'])
        return tracks, artists, date_created

    def _get_data(self, dataset, fma_id, fields=None):
        url = self.BASE_URL + dataset + 's.json?'
        url += dataset + '_id=' + str(fma_id) + '&api_key=' + self.api_key
        # print(url)
        r = requests.get(url)
        r.raise_for_status()
        if r.json()['errors']:
            raise Exception(r.json()['errors'])
        data = r.json()['dataset'][0]
        r_id = data[dataset + '_id']
        if r_id != str(fma_id):
            raise Exception('The received id {} does not correspond to'
                            'the requested one {}'.format(r_id, fma_id))
        if fields is None:
            return data
        if type(fields) is list:
            ret = {}
            for field in fields:
                ret[field] = data[field]
            return ret
        else:
            return data[fields]

    def get_track(self, track_id, fields=None):
        return self._get_data('track', track_id, fields)

    def get_album(self, album_id, fields=None):
        return self._get_data('album', album_id, fields)

    def get_artist(self, artist_id, fields=None):
        return self._get_data('artist', artist_id, fields)

    def get_all(self, dataset, id_range):
        index = dataset + '_id'

        id_ = 2 if dataset is 'track' else 1
        row = self._get_data(dataset, id_)
        df = pd.DataFrame(columns=row.keys())
        df.set_index(index, inplace=True)

        not_found_ids = []

        for id_ in id_range:
            try:
                row = self._get_data(dataset, id_)
            except:
                not_found_ids.append(id_)
                continue
            row.pop(index)
            df.loc[id_] = row

        return df, not_found_ids

    def download_track(self, track_file, path):
        url = 'https://files.freemusicarchive.org/' + track_file
        r = requests.get(url, stream=True)
        r.raise_for_status()
        with open(path, 'wb') as f:
            shutil.copyfileobj(r.raw, f)

    def get_track_genres(self, track_id):
        genres = self.get_track(track_id, 'track_genres')
        genre_ids = []
        genre_titles = []
        for genre in genres:
            genre_ids.append(genre['genre_id'])
            genre_titles.append(genre['genre_title'])
        return genre_ids, genre_titles

    def get_all_genres(self):
        df = pd.DataFrame(columns=['genre_parent_id', 'genre_title',
                                   'genre_handle', 'genre_color'])
        df.index.rename('genre_id', inplace=True)

        page = 1
        while True:
            url = self.BASE_URL + 'genres.json?limit=50'
            url += '&page={}&api_key={}'.format(page, self.api_key)
            r = requests.get(url)
            for genre in r.json()['dataset']:
                genre_id = int(genre.pop(df.index.name))
                df.loc[genre_id] = genre
            assert (r.json()['page'] == str(page))
            page += 1
            if page > r.json()['total_pages']:
                break

        return df


class Genres:

    def __init__(self, genres_df):
        self.df = genres_df

    def create_tree(self, roots, depth=None):

        if type(roots) is not list:
            roots = [roots]
        graph = pydot.Dot(graph_type='digraph', strict=True)

        def create_node(genre_id):
            title = self.df.at[genre_id, 'title']
            ntracks = self.df.at[genre_id, '#tracks']
            #name = self.df.at[genre_id, 'title'] + '\n' + str(genre_id)
            name = '"{}\n{} / {}"'.format(title, genre_id, ntracks)
            return pydot.Node(name)

        def create_tree(root_id, node_p, depth):
            if depth == 0:
                return
            children = self.df[self.df['parent'] == root_id]
            for child in children.iterrows():
                genre_id = child[0]
                node_c = create_node(genre_id)
                graph.add_edge(pydot.Edge(node_p, node_c))
                create_tree(genre_id, node_c,
                            depth-1 if depth is not None else None)

        for root in roots:
            node_p = create_node(root)
            graph.add_node(node_p)
            create_tree(root, node_p, depth)

        return graph

    def find_roots(self):
        roots = []
        for gid, row in self.df.iterrows():
            parent = row['parent']
            title = row['title']
            if parent == 0:
                roots.append(gid)
            elif parent not in self.df.index:
                msg = '{} ({}) has parent {} which is missing'.format(
                        gid, title, parent)
                raise RuntimeError(msg)
        return roots


def load(filepath):

    filename = os.path.basename(filepath)

    if 'features' in filename:
        return pd.read_csv(filepath, index_col=0, header=[0, 1, 2])

    if 'echonest' in filename:
        return pd.read_csv(filepath, index_col=0, header=[0, 1, 2])

    if 'genres' in filename:
        return pd.read_csv(filepath, index_col=0)

    if 'tracks' in filename:
        tracks = pd.read_csv(filepath, index_col=0, header=[0, 1])

        COLUMNS = [('track', 'tags'), ('album', 'tags'), ('artist', 'tags'),
                   ('track', 'genres'), ('track', 'genres_all')]
        for column in COLUMNS:
            tracks[column] = tracks[column].map(ast.literal_eval)

        COLUMNS = [('track', 'date_created'), ('track', 'date_recorded'),
                   ('album', 'date_created'), ('album', 'date_released'),
                   ('artist', 'date_created'), ('artist', 'active_year_begin'),
                   ('artist', 'active_year_end')]
        for column in COLUMNS:
            tracks[column] = pd.to_datetime(tracks[column])

        SUBSETS = ('small', 'medium', 'large')
        tracks['set', 'subset'] = tracks['set', 'subset'].astype(
                pd.CategoricalDtype(categories=SUBSETS, ordered=True))
        #tracks['set', 'subset'] = tracks['set', 'subset'].cat.reorder_categories(SUBSETS, inplace=True)

        COLUMNS = [('track', 'genre_top'), ('track', 'license'),
                   ('album', 'type'), ('album', 'information'),
                   ('artist', 'bio')]
        for column in COLUMNS:
            tracks[column] = tracks[column].astype('category')
            tracks[column].cat.add_categories('MISSING', inplace=True)
            tracks[column].fillna('MISSING', inplace=True)

        return tracks


def get_audio_path(audio_dir, track_id, extension="mp3"):
    tid_str = '{:06d}'.format(track_id)
    return os.path.join(audio_dir, tid_str[:3], tid_str + f'.{extension}')


class Loader:
    def load(self, filepath):
        raise NotImplemented()


class RawAudioLoader(Loader):
    def __init__(self, sampling_rate=SAMPLING_RATE):
        self.sampling_rate = sampling_rate
        self.shape = (NB_AUDIO_SAMPLES * sampling_rate // SAMPLING_RATE, )

    def load(self, filepath):
        return self._load(filepath)[:self.shape[0]]


class LibrosaLoader(RawAudioLoader):
    def _load(self, filepath):
        import librosa
        sr = self.sampling_rate if self.sampling_rate != SAMPLING_RATE else None
        x, sr = librosa.load(filepath, sr=sr)
        return x


class AudioreadLoader(RawAudioLoader):
    def _load(self, filepath):
        import audioread
        a = audioread.audio_open(filepath)
        a.read_data()


class PydubLoader(RawAudioLoader):
    def _load(self, filepath):
        from pydub import AudioSegment
        song = AudioSegment.from_file(filepath)
        song = song.set_channels(1)
        x = song.get_array_of_samples()
        return np.array(x)



class FeatureLoader(object):
    def __init__(self, features):
        self.features = features
        self.shape = (features.shape[1],)

    def load(self, tid):
        return self.features.loc[tid].to_numpy()
        #return np.pad(self.features.loc[tid].to_numpy(), self.shape[0])

class FfmpegLoader(RawAudioLoader):
    def _load(self, filepath):
        """Fastest and less CPU intensive loading method."""
        import subprocess as sp
        with open(filepath.replace('mp3', 'fmpg'), 'rb') as f:
            return np.frombuffer(f.read(), dtype="int16")


def build_sample_loader(audio_dir, Y, loader, extension="mp3", loader_type="filepath"):

    class SampleLoader:

        def __init__(self, tids, batch_size=4):
            self.lock1 = multiprocessing.Lock()
            self.lock2 = multiprocessing.Lock()
            self.batch_foremost = sharedctypes.RawValue(ctypes.c_int, 0)
            self.batch_rearmost = sharedctypes.RawValue(ctypes.c_int, -1)
            self.condition = multiprocessing.Condition(lock=self.lock2)

            data = sharedctypes.RawArray(ctypes.c_int, tids) #.data
            self.tids = np.ctypeslib.as_array(data)

            self.batch_size = batch_size
            self.loader = loader
            self.X = np.empty((self.batch_size, *loader.shape))
            self.Y = np.empty((self.batch_size, Y.shape[1]), dtype=np.int)

        def __iter__(self):
            return self

        def __next__(self):

            with self.lock1:
                if self.batch_foremost.value == 0:
                    np.random.shuffle(self.tids)

                batch_current = self.batch_foremost.value
                if self.batch_foremost.value + self.batch_size < self.tids.size:
                    batch_size = self.batch_size
                    self.batch_foremost.value += self.batch_size
                else:
                    batch_size = self.tids.size - self.batch_foremost.value
                    self.batch_foremost.value = 0

                tids = np.array(self.tids[batch_current:batch_current+batch_size])
            file = open('bad_tid.txt', 'a')    
            bad_tids = [5, 140, 141, 10]
            for i, tid in enumerate(tids):

                # try:
                if loader_type == "filepath":
                    self.X[i] = self.loader.load(get_audio_path(audio_dir, tid, extension=extension))
                else:
                    self.X[i] = self.loader.load(tid)
                self.Y[i] = Y.loc[tid]
                if tid in bad_tids:
                    continue
                try:
                    ffmpegout = self.loader.load(get_audio_path(audio_dir, tid, extension=extension))
                    self.X[i] = ffmpegout
                    self.Y[i] = Y.loc[tid]
                except Exception as e:
                    print('\nException raised while trying to load track for tid {tid}')
                    print(str(e))
                                         
                    file.write(str(tid) + "\n")                       
            file.close() 

            with self.lock2:
                while (batch_current - self.batch_rearmost.value) % self.tids.size > self.batch_size:
                    self.condition.wait()
                self.condition.notify_all()
                self.batch_rearmost.value = batch_current
                return self.X[:batch_size], self.Y[:batch_size]

    return SampleLoader
