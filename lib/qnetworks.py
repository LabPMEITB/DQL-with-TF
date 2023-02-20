from tensorflow.keras import Model
from tensorflow.keras import Input
from tensorflow.keras.layers import Dense

class QNetwork(Model):
    def __init__(self, n_actions, h_layers=None):
        super(QNetwork, self).__init__()
        # Initialize hidden layers
        if h_layers is None:
            self.hidden_layers = [(Dense(20, activation='relu', name='Hidden'))]
        else:
            self.hidden_layers = h_layers

		# Initialize output layer
        self.output_layer = Dense(n_actions, activation=None, name='Output')

    def call(self, inputs):
        x = inputs
        for layer in self.hidden_layers:
            x = layer(x)
        return self.output_layer(x)
    
    def build_graph(self, n_states, name=None):
        x = Input(shape=(n_states, ), ragged=True, name='Input')
        return Model(inputs=[x], outputs=self.call(x), name=name)
