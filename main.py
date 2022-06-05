from AnalyseMelody import *
from DataLoadProcess import *
from RNNLyrics import RNNLyrics
from RNNLyricsMelody import RNNLyricsMelody
from keras.callbacks import TensorBoard
from datetime import datetime
import matplotlib.pyplot as plt
from sklearn.metrics.pairwise import cosine_similarity
import pickle
import os
import numpy as np
import random
import pandas as pd


''' Paths'''
OUTPUT_PATH = 'output/'
GLOVE_PATH = 'input/glove.6B.300d.txt'
MIDI_PATH = 'midi_files/'
TRAIN_PATH = 'input/lyrics_train_set.csv'
TEST_PATH = 'input/lyrics_test_set.csv'
MIDI_PATH_TRAIN_PICKLE = 'output/midi_train_pkl.pkl'
MIDI_PATH_TEST_PICKLE = 'output/midi_test_pkl.pkl'
LYRICS_PATH_TRAIN_PICKLE = 'output/lyrics_final_train_pkl.pkl'
LYRICS_PATH_TEST_PICKLE = 'output/lyrics_final_test_pkl.pkl'
MELODIES_PATH_TEST_PICKLE = 'output/melodies_test_pkl.pkl'
MELODIES_PATH_TRAIN_PICKLE = 'output/melodies_train_pkl_1_a.pkl'
MELODIES_PATH_VAL_PICKLE = 'output/melodies_val_pkl_1_a.pkl'
MELODIES_PATH_TEST_PICKLE = 'output/melodies_test_pkl_1_a.pkl'
MELODIES_PATH_TRAIN_PICKLE_2 = 'output/melodies_train_pkl_2.pkl'
MELODIES_PATH_VAL_PICKLE_2 = 'output/melodies_val_pkl_2.pkl'
MELODIES_PATH_TEST_PICKLE_2 = 'output/melodies_test_pkl_2.pkl'
WORD2VEC_PATH_PICKLE = 'output/word2vec_pkl.pkl'
WORD2VEC_MAT_PATH_PICKLE = 'output/word2vec_mat_pkl.pkl'

''' Constants'''
SEQ_LENGTH = 5
TRAIN_SIZE = 0.8
GLOVE_SIZE = 300
LEARNING_RATE = 0.00001
EPOCHS = 50
BATCH_SIZE = 128
UNITS = 256


def run_experiment(model_name, method):
    # create the output folder
    if not os.path.exists(OUTPUT_PATH):
        os.makedirs(OUTPUT_PATH, mode=0o777)

    # load train pickle objects
    if os.path.exists(MIDI_PATH_TRAIN_PICKLE):
        with open(MIDI_PATH_TRAIN_PICKLE, 'rb') as midi_input_train:
            songs_train = pickle.load(midi_input_train)
        with open(LYRICS_PATH_TRAIN_PICKLE, 'rb') as train_lyrics_input:
            lyrics_final_train = pickle.load(train_lyrics_input)
    else:
        # load train data
        l1_train, l2_train, l3_train = load_input_files(TRAIN_PATH)
        songs_train, lyrics_final_train = load_midi_files(l1_train, l2_train, l3_train, MIDI_PATH)
        with open(MIDI_PATH_TRAIN_PICKLE, 'wb') as midi_output_train:
            pickle.dump(songs_train, midi_output_train, pickle.HIGHEST_PROTOCOL)
        with open(LYRICS_PATH_TRAIN_PICKLE, 'wb') as train_lyrics_outputs:
            pickle.dump(lyrics_final_train, train_lyrics_outputs, pickle.HIGHEST_PROTOCOL)

    # load test pickle objects
    if os.path.exists(MIDI_PATH_TEST_PICKLE):
        with open(MIDI_PATH_TEST_PICKLE, 'rb') as midi_input_test:
            songs_test = pickle.load(midi_input_test)
        with open(LYRICS_PATH_TEST_PICKLE, 'rb') as test_lyrics_input:
            lyrics_final_test = pickle.load(test_lyrics_input)
    else:
        # load test data
        l1_test, l2_test, l3_test = load_input_files(TEST_PATH)
        songs_test, lyrics_final_test = load_midi_files(l1_test, l2_test, l3_test, MIDI_PATH)
        with open(MIDI_PATH_TEST_PICKLE, 'wb') as midi_output_test:
            pickle.dump(songs_test, midi_output_test, pickle.HIGHEST_PROTOCOL)
        with open(LYRICS_PATH_TEST_PICKLE, 'wb') as test_lyrics_outputs:
            pickle.dump(lyrics_final_test, test_lyrics_outputs, pickle.HIGHEST_PROTOCOL)

    # prepare the entire lyrics dictionaries
    lyrics_final_all = lyrics_final_train + lyrics_final_test
    word2index, index2word = create_word_to_index(lyrics_final_all)

    # load word2vec pickle
    if os.path.exists(WORD2VEC_PATH_PICKLE):
        with open(WORD2VEC_PATH_PICKLE, 'rb') as word2vec_input:
            vec = pickle.load(word2vec_input)
    else:
        # create word2vec dictionary
        vec = get_word2vec(GLOVE_PATH, GLOVE_SIZE)
        with open(WORD2VEC_PATH_PICKLE, 'wb') as word2vec_output:
            pickle.dump(vec, word2vec_output, pickle.HIGHEST_PROTOCOL)

    # load word2vec matrix pickle
    if os.path.exists(WORD2VEC_MAT_PATH_PICKLE):
        with open(WORD2VEC_MAT_PATH_PICKLE, 'rb') as word2vec_mat_output:
            word2vec_matrix = pickle.load(word2vec_mat_output)
    else:
        # create word2vec matrix pickle
        word2vec_matrix = get_word2vec_matrix(word2index, vec)
        with open(WORD2VEC_MAT_PATH_PICKLE, 'wb') as word2vec_mat_output:
            pickle.dump(word2vec_matrix, word2vec_mat_output, pickle.HIGHEST_PROTOCOL)

    model = None
    if model_name == 'lyrics':
        # prepare all sets
        x_train, y_train, x_val, y_val, x_test, y_test = prepare_all_sets\
            (lyrics_final_train, lyrics_final_test, SEQ_LENGTH, TRAIN_SIZE, word2index)

        # create model
        model = RNNLyrics()
        model.build_network(word2vec_matrix, SEQ_LENGTH, GLOVE_SIZE, LEARNING_RATE, UNITS)
        print(model.summary())

        tensor_board_cb = TensorBoard('logs/test_songs', update_freq=1)
        print('train model..')
        train_history = model.fit(x_train=x_train, y_train=y_train, epochs=EPOCHS, batch_size=BATCH_SIZE,
                                  validation_data=(x_val, y_val), callbacks=[tensor_board_cb])
        # model.load_model('model_RNNLyricsMelody_05_18_2021_16_40_57')

    elif model_name == 'lyrics_melody':
        # prepare all sets
        x_train, y_train, x_val, y_val, x_test, y_test, train_melodies, val_melodies, train_lyrics, val_lyrics = \
            prepare_all_sets(lyrics_final_train, lyrics_final_test, SEQ_LENGTH, TRAIN_SIZE, word2index, songs_train)
        number_of_all_seq_train = x_train.shape[0]
        number_of_all_seq_val = x_val.shape[0]

        # create model
        model = RNNLyricsMelody()
        if method == 1:
            model.build_network(word2vec_matrix, SEQ_LENGTH, GLOVE_SIZE, LEARNING_RATE, UNITS, 3)
            train_path = MELODIES_PATH_TRAIN_PICKLE
            val_path = MELODIES_PATH_VAL_PICKLE
            test_path = MELODIES_PATH_TEST_PICKLE
            func = extract_all_features_method1_a
        elif method == 2:
            model.build_network(word2vec_matrix, SEQ_LENGTH, GLOVE_SIZE, LEARNING_RATE, UNITS, 128)
            train_path = MELODIES_PATH_TRAIN_PICKLE_2
            val_path = MELODIES_PATH_VAL_PICKLE_2
            test_path = MELODIES_PATH_TEST_PICKLE_2
            func = extract_all_features_method2
        print(model.summary())
        # load melodies features - train
        if os.path.exists(train_path):
            print('load melodies features- train..')
            with open(train_path, 'rb') as midi_features_train:
                train_melody_features = pickle.load(midi_features_train)
        else:
            # create melodies features
            print('create melodies features- train..')
            train_melody_features = func(train_melodies, train_lyrics, SEQ_LENGTH,
                                         number_of_all_seq_train)
            with open(train_path, 'wb') as midi_features_train:
                pickle.dump(train_melody_features, midi_features_train, pickle.HIGHEST_PROTOCOL)

        # load melodies features - validation
        if os.path.exists(val_path):
            print('load melodies features- val..')
            with open(val_path, 'rb') as midi_features_val:
                val_melody_features = pickle.load(midi_features_val)
        else:
            # create melodies features
            print('create melodies features- val..')
            val_melody_features = func(val_melodies, val_lyrics, SEQ_LENGTH, number_of_all_seq_val)
            with open(val_path, 'wb') as midi_features_val:
                pickle.dump(val_melody_features, midi_features_val, pickle.HIGHEST_PROTOCOL)

        # load melodies features - test
        if os.path.exists(test_path):
            print('load melodies features- test..')
            with open(test_path, 'rb') as midi_features_test:
                test_melody_features = pickle.load(midi_features_test)
        else:
            # create melodies features
            print('create melodies features- test..')
            test_melodies = np.array(songs_test)
            test_lyrics = np.array(lyrics_final_test)
            number_of_all_seq_test = x_test.shape[0]
            test_melody_features = func(test_melodies, test_lyrics, SEQ_LENGTH, number_of_all_seq_test)
            with open(test_path, 'wb') as midi_features_test:
                pickle.dump(test_melody_features, midi_features_test, pickle.HIGHEST_PROTOCOL)

        x_train = [x_train, train_melody_features]
        x_val = [x_val, val_melody_features]
        x_test = [x_test, test_melody_features]

        tensor_board_cb = TensorBoard('logs/test_songs_1a', update_freq=1)
        print('train model..')
        train_history = model.fit(x_train=x_train, y_train=y_train, epochs=EPOCHS, batch_size=BATCH_SIZE,
                                  validation_data=(x_val, y_val), callbacks=[tensor_board_cb])
        # model.load_model('model_RNNLyricsMelody_05_18_2021_20_11_35')

    print('finished training model..')
    dt = datetime.now().strftime('%m_%d_%Y_%H_%M_%S')
    model.save_model(dt)
    print('the model was saved..')
    df_results = plot_acc_loss(train_history, dt)

    # evaluate
    loss, accuracy = model.evaluate(x_test, y_test, BATCH_SIZE)
    print(f'test accuracy: {accuracy}, test loss: {loss}')
    df_results['test accuracy'] = [accuracy]
    df_results['test loss'] = [loss]
    df_results['learning rate'] = [LEARNING_RATE]
    df_results['epochs'] = [EPOCHS]
    df_results['batch size'] = [BATCH_SIZE]
    df_results['seq length'] = [SEQ_LENGTH]
    df_results.to_csv('results_' + dt + '.csv', index=False)

    test_x_split = [lyrics.split(' ') for lyrics in lyrics_final_test]
    test_x_to_predict = [lyric[0] for lyric in test_x_split]
    test_y_to_predict = [lyric[1:] for lyric in test_x_split]
    test_song_lengths = [len(test_x_split_i) for test_x_split_i in test_x_split]

    # predict
    print('predicting..')
    end = 0
    seeds = [test_x_to_predict[0], test_x_to_predict[1], test_x_to_predict[2]]
    for i in range(len(test_x_to_predict)):
        y_true = [word2index[y_true_i] for y_true_i in test_y_to_predict[i]]
        if model_name == 'lyrics':
            predict_one_song(model, test_x_to_predict[i], SEQ_LENGTH, index2word, word2index, y_true, word2vec_matrix, dt)
        elif model_name == 'lyrics_melody':
            seq_count = test_song_lengths[i] - SEQ_LENGTH
            start = end
            end = start + seq_count
            for seed in seeds:
                predict_one_song(model, seed, SEQ_LENGTH, index2word, word2index, y_true, word2vec_matrix, dt,
                                 test_melody_features[start:end, :, :])


def plot_acc_loss(history, dt):
    """
    This function plots the loss and accuracy graphs
    :param history: the model's training history
    :param dt: the data and time string to mark the file
    :return: a dataframe with the loss and accuracy of the train and validation sets of the last epoch
    """
    plt.plot(history.history['accuracy'])
    plt.plot(history.history['val_accuracy'])
    plt.title('model accuracy')
    plt.ylabel('accuracy')
    plt.xlabel('epoch')
    plt.legend(['train', 'validation'], loc='upper left')
    plt.savefig(f'figs/accuracy_{dt}.png')
    plt.clf()
    # summarize history for loss
    plt.plot(history.history['loss'])
    plt.plot(history.history['val_loss'])
    plt.title('model loss')
    plt.ylabel('loss')
    plt.xlabel('epoch')
    plt.legend(['train', 'validation'], loc='upper left')

    if not os.path.exists('figs/'):
        os.makedirs('figs/')
    plt.savefig(f'figs/loss_{dt}.png')
    plt.clf()

    df_res = pd.DataFrame({
        'last train loss': [history.history['loss'][-1]],
        'last val loss': [history.history['val_loss'][-1]],
        'last train accuracy': [history.history['accuracy'][-1]],
        'last val accuracy': [history.history['val_accuracy'][-1]]
    })
    return df_res


def select_next_word_2(word_probs, index2word):
    """
    A method for selecting the next word based on the probabilities the model returned as the softmax output.
    The likelihood of a term to be selected by the sampling is proportional to its probability.
    :param word_probs: softmax output, a probability for each word in the vocabulary
    :param index2word:  the dictionary that maps an index to a word in the vocabulary
    :return: the predicted word and its index
    """
    vocab_probs = word_probs.T
    word_idx_arr = np.arange(vocab_probs.size)
    word_idx = random.choices(word_idx_arr, k=1, weights=vocab_probs)[0]
    predicted_word = index2word[word_idx]
    return predicted_word, word_idx


def predict_one_song(trained_model, word, seq_length, index2word, word2index, y_true, word2vec_matrix, dt, melody=None):
    """
    This function receives the trained model and input of one word and melody, and generates the rest of the song.
    It uses select_next_word_2 to select the next word based on probabilities. Once the entire lyrics are produced,
    it calculated two metrics: Mean Euclidean distance, Mean Cosine similarity.
    The function writes the predicted lyrics and the evaluation metrics to file.
    :param trained_model: the model trained on the train set
    :param word: the first word to give the model as input for the first sequence
    :param seq_length: the length of one sequence the model was trained on
    :param index2word: the dictionary that maps an index to a word in the vocabulary
    :param word2index: the dictionary that maps a word to its index in the vocabulary
    :param y_true: the original lyrics
    :param word2vec_matrix: the word2vec matrix needed for the evaluation
    :param dt: the date and time of the training
    :param melody: the melody features
    """
    input_lyrics = np.zeros((1, seq_length, 1))  # for one sequence
    predicted_song = [word]
    predicted_song_idx = []
    last_position = seq_length - 1
    input_lyrics[0, last_position, 0] = word2index[word]
    j = 0
    end_song_word = 'ENDSONG'
    end_line_word = 'ENDLINE'
    predicted_word = ''
    if melody is not None:
        melody_concat = melody.reshape((-1, melody.shape[2]))
        melody_pad = np.zeros((melody_concat.shape[0] + (seq_length-1), melody.shape[2]))
        for seq in range(melody_concat.shape[0]):
            melody_pad[seq + (seq_length - 1), :] = melody_concat[seq, :]
        number_of_seq = melody.shape[0] + (seq_length-1)
        melody = np.array([melody_pad[sequence_n: sequence_n + seq_length]
                           for sequence_n in range(number_of_seq)])
    while predicted_word != end_song_word and j < melody.shape[0]:
        if melody is None:
            predictions = trained_model.predict(input_lyrics)
        else:
            seq_melody = melody[j, :, :].reshape(1, seq_length, -1)
            predictions = trained_model.predict([input_lyrics, seq_melody])
        predicted_word, predicted_word_index = select_next_word_2(predictions[0], index2word)
        predicted_song_idx.append(predicted_word_index)
        predicted_song.append(predicted_word)
        # shift one place to the left and insert a new word
        input_lyrics = np.roll(input_lyrics, -1, axis=1)
        input_lyrics[0, last_position, 0] = predicted_word_index
        j += 1

    song = ' '.join(predicted_song).replace(end_line_word, '\n')
    embedding_distance = embedding_distance_all_song(y_true, predicted_song_idx, word2vec_matrix)
    embedding_cossim = embedding_cosine_sim_all_song(y_true, predicted_song_idx, word2vec_matrix)

    f = open(f'results_{dt}.txt', "a")
    f.write(f'seed: {word} \n predicted song: {song}')
    f.write(f'metrics- embedding distance: {np.round(embedding_distance, 4)}, embedding cossim: {np.round(embedding_cossim, 4)} \n')
    f.close()


def embedding_distance_all_song(y_true, y_pred, word2vec_matrix):
    """
    Receives the encoded y_true and y_pred and calculates the euclidean distance between the embedding
    vectors of each work
    :param y_true:
    :param y_pred:
    :param word2vec_matrix:
    :return:
    """
    words_true_enb = [word2vec_matrix[word_true, :] for word_true in y_true]
    words_pred_enb = [word2vec_matrix[word_pred, :] for word_pred in y_pred]
    distances = np.array([np.linalg.norm(word_pred_enb - word_true_enb) for word_true_enb, word_pred_enb
                          in zip(words_true_enb, words_pred_enb)])
    return np.mean(distances)


def embedding_cosine_sim_all_song(y_true, y_pred, word2vec_matrix):
    """
    Receives the encoded y_true and y_pred and calculates the euclidean cosine similarity between the embedding
    vectors of each work
    :param y_true:
    :param y_pred:
    :param word2vec_matrix:
    :return:
    """
    words_true_enb = np.array([word2vec_matrix[word_true, :] for word_true in y_true])
    words_pred_enb = np.array([word2vec_matrix[word_pred, :] for word_pred in y_pred])
    cosine_sim = cosine_similarity(words_pred_enb, words_true_enb)
    return np.mean(cosine_sim)


if __name__ == '__main__':
    model_name_ = 'lyrics_melody'
    method_ = 1
    run_experiment(model_name_, method_)

