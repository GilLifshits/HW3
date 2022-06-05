
from RNN import RNN
from keras.layers import Dense, LSTM, Input, Embedding, Dropout, Bidirectional, Masking
from keras import Sequential
from keras.optimizers import Adam
from keras import Model


class RNNLyrics(RNN):
    """
        A class for the lyrics RNN model
    """
    def __init__(self):
        self.model = None
        self.word2vec_matrix = None

    def build_network(self, word2vec_matrix, seq_length, embedding_size, learning_rate, units):
        self.word2vec_matrix = word2vec_matrix
        vocab_size = word2vec_matrix.shape[0]
        lyrics_input = Input((seq_length,))
        model = Sequential()
        model.add(Embedding(input_dim=vocab_size,
                            input_length=seq_length,
                            output_dim=embedding_size,
                            weights=[word2vec_matrix],
                            trainable=False,
                            mask_zero=True,
                            name='LyricsModel'))
        model.add(Masking(mask_value=0.))
        model.add(Bidirectional(LSTM(units=units, activation='relu')))
        model.add(Dropout(0.2))
        encoded_model = model(lyrics_input)
        pred = Dense(vocab_size, activation='softmax')(encoded_model)

        self.model = Model(inputs=lyrics_input, outputs=pred)
        optimizer = Adam(learning_rate=learning_rate)

        self.model.compile(optimizer=optimizer, loss='categorical_crossentropy',
                           metrics=['accuracy'])
        self.name = 'RNNLyrics'
        print("created model")




