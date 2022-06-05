import keras.utils as utils
from keras.models import model_from_json


class RNN:
    """
    Abstract class of the RNN model
    """
    def __init__(self):
        self.model = None
        self.word2vec_matrix = None
        self.name = ''

    def fit(self, x_train, y_train, epochs, batch_size, validation_data, callbacks):
        return self.model.fit(x=x_train, y=y_train, epochs=epochs, batch_size=batch_size, verbose=1,
                              validation_data=validation_data, callbacks=callbacks)

    def evaluate(self, x_test, y_test, batch_size):
        return self.model.evaluate(x_test, y_test, batch_size)

    def predict(self, t_test):
        return self.model.predict(t_test)

    def save_model(self, dt):
        model_name = self.name
        model_json = self.model.to_json()
        with open(f'model_{model_name}_{dt}.json', "w") as json_f:
            json_f.write(model_json)
        self.model.save_weights(f'model_{model_name}_{dt}.h5')

    def load_model(self, path):
        json_file = open(f'{path}.json', 'r')
        loaded_model_json = json_file.read()
        json_file.close()
        loaded_model = model_from_json(loaded_model_json)
        # load weights into new model
        loaded_model.load_weights(f'{path}.h5')
        print("Loaded model from disk")
        self.model = loaded_model
        return loaded_model

    def summary(self):
        return self.model.summary()

    def plot_model(self, file_name):
        utils.plot_model(
            self.model,
            to_file=file_name,
            show_shapes=True,
            show_layer_names=True,
            rankdir="TB",
            expand_nested=True,
            dpi=96,
        )

