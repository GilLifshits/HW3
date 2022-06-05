from RNN import RNN
from keras.layers import Dense, LSTM, Input, Embedding, Dropout, Bidirectional, Masking, Concatenate
from keras.optimizers import Adam
from keras import Model


class RNNLyricsMelody(RNN):
    """
    A class for the lyrics + melody RNN model
    """
    def __init__(self):
        self.model = None
        self.word2vec_matrix = None

    def build_network(self, word2vec_matrix, seq_length, embedding_size, learning_rate, units, melodies_size):
        self.word2vec_matrix = word2vec_matrix
        vocab_size = word2vec_matrix.shape[0]
        lyrics_input = Input((seq_length,))
        melodies_input = Input((seq_length, melodies_size))
        emb_layer = Embedding(input_dim=vocab_size,
                              input_length=seq_length,
                              output_dim=embedding_size,
                              weights=[word2vec_matrix],
                              trainable=False,
                              mask_zero=True,
                              name='LyricsMelodiesModel')(lyrics_input)
        mask_layer = Masking(mask_value=0.)(emb_layer)
        concat_layer = Concatenate(axis=2)([mask_layer, melodies_input])
        bid_layer = Bidirectional(LSTM(units=units, activation='relu'))(concat_layer)
        dropout_layer = Dropout(0.2)(bid_layer)
        predictions = Dense(vocab_size, activation='softmax')(dropout_layer)

        self.model = Model(inputs=[lyrics_input, melodies_input], outputs=predictions)
        optimizer = Adam(learning_rate=learning_rate)
        self.model.compile(optimizer=optimizer, loss='categorical_crossentropy',
                           metrics=['accuracy'])
        self.name = 'RNNLyricsMelody'
        print("created model")
