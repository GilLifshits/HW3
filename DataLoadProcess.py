"""" Load and preprocess and lyrics and melodies files"""

import pretty_midi
import pandas as pd
import numpy as np
import os
import re
np.random.seed(42)


def load_midi_files(artists, titles, lyrics, midi_files_path):
    """
    This function receives the titles of the songs and their artist, and loads the
    melodies file for the songs. A mid file name is structure as follows:
    'artist_name_-_song_title.mid'.
    :param titles: songs titles
    :param artists: songs artists
    :param lyrics: the songs lyrics
    :param midi_files_path: the path for the midi files
    :return: pm_songs- a list of melodies
    """
    assert len(titles) == len(artists)
    paths_dict = get_all_melodies_dict(midi_files_path)
    pm_songs = []
    lyrics_final = []
    for artist, title, song_lyrics in zip(artists, titles, lyrics):
        song_name = f'{artist}_-_{title}.mid'.replace(" ", "_")
        # get path from the dict
        if song_name not in paths_dict:
            print(f'no melody for song {title}')
            continue
        song_path = paths_dict[song_name]
        try:
            pm = pretty_midi.PrettyMIDI(os.path.join(midi_files_path, song_path))
            pm_songs.append(pm)
            lyrics_final.append(song_lyrics)
        except Exception as e:
            print(f'{song_path} raise {str(e)} an exception from PrettyMIDI')
    return pm_songs, lyrics_final


def load_input_files(input_file_path):
    """
    This function loads the train and test files of the lyrics
    :param input_file_path: the path of the input file
    :return: lists of artists, titles, lyrics
    """
    songs_df = pd.read_csv(input_file_path, sep='\n', header=None)
    songs = songs_df.iloc[:, 0].str.replace(' &', ' ENDLINE').str.extract(r'([^,]+),([^,]+),(.+)')
    songs.columns = ['artist', 'title', 'lyrics']
    artists = songs['artist'].tolist()
    titles = songs['title'].tolist()
    lyrics = songs['lyrics'].to_list()
    artists = [artist.strip(' ') for artist in artists]
    titles = [title.strip(' ') for title in titles]
    lyrics = [" ".join(lyric.split()) for lyric in lyrics]
    punctuations = r"""!"#$%&()*+,-./:;<=>?@[\]^_`{|}~"""
    lyrics = [lyric.translate(str.maketrans(punctuations, ' '*len(punctuations))) for lyric in lyrics]
    lyrics = [" ".join(lyric.split()) for lyric in lyrics]
    lyrics = [clean_text(lyric) + ' ENDSONG' for lyric in lyrics]
    return artists, titles, lyrics


def clean_text(text):
    """
    This function cleans the lyrics text.
    :param text: lyrics text
    :return: the lyrics text after being cleaned.
    """
    text = re.sub(r"\'ve", ' have', text)
    text = re.sub(r"\'ll", ' will', text)
    text = re.sub(r"in'", 'ing', text)
    text = re.sub(r"won't", 'will not', text)
    text = re.sub(r"i'm", 'i am', text)
    text = re.sub(r"he's", 'he is', text)
    text = re.sub(r"she's", 'she is', text)
    text = re.sub(r"it's", 'it is', text)
    text = re.sub(r"we're", 'we are', text)
    text = re.sub(r"you're", 'you are', text)
    text = re.sub(r"they're", 'they are', text)
    text = re.sub(r"who'se", 'who is', text)
    text = re.sub(r"who're", 'who are', text)
    text = re.sub(r"what's", 'what is', text)
    text = re.sub(r"where's", 'where is', text)
    text = re.sub(r"y'all", 'you all', text)
    text = re.sub(r"\'d", ' would', text)
    text = re.sub(r"ain't", 'are not', text)
    text = re.sub(r"can't", 'can not', text)
    text = re.sub(r"evry", 'every', text)
    text = re.sub(r"n't", 'not', text)
    text = re.sub(r"\'s", '', text)
    text = re.sub(r"\'", '', text)
    text = re.sub(r"hasnot", 'has not', text)
    text = re.sub(r"doesnot", 'does not', text)
    text = re.sub(r"couldnot", 'could not', text)
    text = re.sub(r"wouldnot", 'would not', text)
    text = re.sub(r"isnot", 'is not', text)
    text = re.sub(r"havenot", 'have not', text)
    text = re.sub(r"shouldnot", 'have not', text)
    text = re.sub(r"donot", 'do not', text)
    text = re.sub(r"arenot", 'are not', text)
    text = re.sub(r"wasnot", 'was not', text)
    return text


def create_word_to_index(lyrics):
    """
    This function created a dictionary that maps the unique words from all songs to a unique index
    :param lyrics: the lyrics of all songs
    :return: a word_to_index dictionary
    """
    unique_words = list(set(" ".join(lyrics).split(" ")))
    unique_words.sort()
    word2index = {word: idx for idx, word in enumerate(unique_words)}
    index2word = {idx: word for idx, word in enumerate(unique_words)}
    return word2index, index2word


def get_word2vec(word2vec_path, vector_size, encoding='utf-8'):
    """
    This function gets the word2vec embeddings file and create a dictionary that maps the word
    to her vector off embedding values.
    :return: word2vec_dict
    """
    word2vec_dict = {}
    with open(word2vec_path, 'r', encoding=encoding) as f:
        word2vec_lines = f.readlines()
    for line in word2vec_lines:
        split_line = line.split(' ')
        word = split_line[0]
        embedding = split_line[1:]
        assert len(embedding) == vector_size
        word2vec_dict[word] = [float(embedding_) for embedding_ in embedding]
    avg_vec = add_average_vector_to_word2vec(word2vec_path, len(word2vec_lines), vector_size)
    avg_vec_end_line = avg_vec * 5
    avg_vec_end_song = avg_vec * 6
    word2vec_dict['notIncluded'] = avg_vec
    word2vec_dict['ENDLINE'] = avg_vec_end_line
    word2vec_dict['ENDSONG'] = avg_vec_end_song
    return word2vec_dict


def add_average_vector_to_word2vec(word2vec_path, dict_length, vector_size, encoding='utf-8'):
    """
    This function create a vector of average embedding values for all the word that are not in the word2vec of glove.
    :param word2vec_path:
    :param dict_length:
    :param vector_size: the glove vector size
    :param encoding:
    :return: a vector of average values
    """
    n_vec = dict_length
    hidden_dim = vector_size
    vecs = np.zeros((n_vec, hidden_dim), dtype=np.float32)

    with open(word2vec_path, 'r', encoding=encoding) as f:
        for i, line in enumerate(f):
            vecs[i] = np.array([float(n) for n in line.split(' ')[1:]], dtype=np.float32)

    average_vec = np.mean(vecs, axis=0)
    return average_vec


def get_all_melodies_dict(midi_folder_path):
    """
    This function maps the lower case name of the song file to the sentence case name
    :param midi_folder_path: the path of the midi folder
    :return: midi_dict - a dictionary with the mapping of names,
    where the key is the name in lower case letters (as appear in the train/test files)
    and the value is the original path of the midi file.
    """
    midi_dict = {}
    for filename in os.listdir(midi_folder_path):
        if filename.endswith('.mid'):
            file_name_split = filename.split('-')
            filename_key = "-_".join([file_name_split[0], file_name_split[1].strip('_')])
            if not filename_key.endswith('.mid'):
                filename_key += '.mid'
            midi_dict[filename_key.lower()] = filename
    return midi_dict


def get_word2vec_matrix(word2index, word2vec_dict):
    """
    This function creates a matrix of word2vec embeddings for all the unique words in all songs
    using the word2vec_dict. If a word does not appear in the glove file it will not be in the matrix
    :param word2index: mapping dictionary for all words in all songs and a unique index
    :param word2vec_dict: the dictionary of all the words and their embeddings from the glove
    :return: word2vec_matrix
    """
    word2vec_matrix = []
    not_exist_counter = 0
    for word, word_idx in word2index.items():
        if word not in word2vec_dict:
            not_exist_counter += 1
            # print(f"the word {word} is not in word2vec_dict")
            word2vec_matrix.append(np.array(word2vec_dict['notIncluded']))
        else:
            word2vec_matrix.append(np.array(word2vec_dict[word]))
    print(f"{not_exist_counter} words were not found")
    return np.array(word2vec_matrix)
    

def one_hot_encoding(words_to_predict, word2index):
    """
    This function creates the one hot encoding matrix for the predicted words (y).
    It receives the words the model has to predict (the next word of each sequence)
    and the word2index mapping dictionary and returns a matrix each row is the one hot encoding
    of one predicted word. The matrix is of size(amount of predicted words, total amount of unique
    words in all songs).
    :param words_to_predict: a list of the next word of each sequence
    :param word2index: mapping dictionary for all words in all songs and a unique index
    :return: one hot encoded matrix
    """
    n_total_words = len(word2index)
    encoded_matrix = np.zeros((len(words_to_predict), n_total_words))
    for predicted_word_idx, predicted_word in enumerate(words_to_predict):
        word_encoded_idx = word2index[predicted_word]
        encoded_matrix[predicted_word_idx, word_encoded_idx] = 1
    return encoded_matrix


def index_encoding(sequence, word2index):
    """
    This function receives a sequence and the word2index dictionary and returns
    the sequence encoded by indexes
    :param sequence: a list of words
    :param word2index: mapping dictionary for all words in all songs and a unique index
    :return: a list of the indexes of the words from the word2index
    """
    encoded_list = [word2index[word] for word in sequence]
    return encoded_list


def split_to_sequences(seq_length, song_lyrics, word2index):
    """
    This function splits a single song into sequences.
    It receives a string contains the song's lyrics, sequence length and a dictionary maps
    each word into unique index.
    The function encodes the predicted word of each sequence using the one hot encoder.
    :param seq_length: sequences length
    :param song_lyrics: string contains the song's lyrics
    :param word2index: a dictionary that maps each word in all songs to a unique index
    :return: sequences: all the sequences of the given lyrics
             one_hot_encoded_y: a matrix which contains the one hot encoding for each "predicted" word
             corresponding to the sequences.
    """
    song_lyrics_split = song_lyrics.split(" ")
    sequences = []
    words_to_predict = []
    for seq_idx in range(len(song_lyrics_split) - seq_length):
        start = seq_idx
        end = seq_idx + seq_length
        sequence = song_lyrics_split[start: end]
        encoded_sequence = index_encoding(sequence, word2index)
        word_to_predict = song_lyrics_split[end]
        sequences.append(encoded_sequence)
        words_to_predict.append(word_to_predict)
    one_hot_encoded_y = one_hot_encoding(words_to_predict, word2index)
    return np.array(sequences), one_hot_encoded_y


def create_sequences_for_all_songs(seq_length, all_songs_lyrics, word2index):
    """
    This function iterates over all the songs and create the sequences and one_hot_encoded_y
    for every song
    :param seq_length: the length of one sequence
    :param all_songs_lyrics: the array of all songs
    :param word2index: a dictionary that maps each word in all songs to a unique index
    :return: x, y
    """
    x, y = [], []
    for song in all_songs_lyrics:
        sequences, one_hot_encoded_y = split_to_sequences(seq_length, song, word2index)
        x.extend(sequences)
        y.extend(one_hot_encoded_y)
    return np.array(x), np.array(y)
    

def split_to_train_validation(train_percentage, lyrics_data, melody_data=None):
    """
    This function receives the lyrics and melody of all training songs.
    It splits the data into training and validation sets
    :param train_percentage: the proportion of data to reserve for the train
    :param lyrics_data: a list of strings representing the lyrics of all songs
    :param melody_data: a list of mid files, corresponding to the lyrics, None if not received
    :return: list of training lyrics
            list of training melodies
            list of validation lyrics - None if the melodies was None
            list of validation melodies - None if the melodies was None
    """
    all_indexes = np.arange(len(lyrics_data))
    np.random.shuffle(all_indexes)
    train_size = int(len(lyrics_data)*train_percentage)
    train_indexes = all_indexes[0: train_size]
    val_indexes = all_indexes[train_size:]
    train_lyrics = np.array(lyrics_data)[train_indexes]
    val_lyrics = np.array(lyrics_data)[val_indexes]
    train_melodies, val_melodies = None, None
    if melody_data is not None:
        train_melodies = np.array(melody_data)[train_indexes]
        val_melodies = np.array(melody_data)[val_indexes]
    return train_lyrics, val_lyrics, train_melodies, val_melodies


def prepare_all_sets(train_val_lyrics, test_lyrics, seq_length, train_percentage, word2index,
                     train_val_melodies=None):
    """
    This function receives the train and test lyrics and melodies. 
    The train is split into train and validation sets. All sets are divided into 
    x and y, when x is sequences of all lyrics and y is the one hot encoded matrices of the "predicted" 
    words. 
    :param train_val_lyrics: list of strings of all train songs lyrics 
    :param test_lyrics: list of strings of all test songs lyrics 
    :param seq_length: the size of the sequence
    :param train_percentage: 
    :param word2index: a dictionary that maps each word in all songs to a unique index
    :param train_val_melodies: list of the melodies in the train and validation sets (before split) 
    :return: all the X and Y sets
    """
    train_lyrics, val_lyrics, train_melodies, val_melodies = split_to_train_validation\
        (train_percentage, train_val_lyrics, train_val_melodies)
    train_x, train_y = create_sequences_for_all_songs(seq_length, train_lyrics, word2index)
    val_x, val_y = create_sequences_for_all_songs(seq_length, val_lyrics, word2index)

    test_x, test_y = create_sequences_for_all_songs(seq_length, test_lyrics, word2index)
    if train_melodies is None:
        return train_x, train_y, val_x, val_y, test_x, test_y
    else:
        return train_x, train_y, val_x, val_y, test_x, test_y, train_melodies, val_melodies, train_lyrics, val_lyrics
