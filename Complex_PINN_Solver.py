import numpy as np
import tensorflow as tf
from tensorflow.keras.layers import Dense
from tensorflow.keras import Model
from scipy import special

class ComplexPINN(Model):
    def __init__(self, layers):
        super(ComplexPINN, self).__init__()
        self.hidden = [Dense(units=units, activation='tanh') for units in layers[1:-1]]
        self.out = Dense(units=layers[-1])

    def call(self, x):
        for layer in self.hidden:
            x = layer(x)
        output = self.out(x)
        return output

def train_step(model, optimizer, t, x, u_real, u_imag):
    with tf.GradientTape(persistent=True) as tape:
        tape.watch(t)
        tape.watch(x)
        outputs = model(tf.concat([t, x], 1))
        u_pred_real = outputs[:, 0:1]
        u_pred_imag = outputs[:, 1:2]
        u_pred = tf.complex(u_pred_real, u_pred_imag)

        u_pred_t = tape.gradient(u_pred, t)
        u_pred_x = tape.gradient(u_pred, x)
        u_pred_xx = tape.gradient(u_pred_x, x)
        loss = tf.reduce_mean(tf.square(u_real - u_pred_real)) + \
               tf.reduce_mean(tf.square(u_imag - u_pred_imag)) + \
               tf.reduce_mean(tf.square(u_pred_t - 0.5 * u_pred_xx - tf.square(tf.abs(u_pred)) * u_pred))
    gradients = tape.gradient(loss, model.trainable_variables)
    optimizer.apply_gradients(zip(gradients, model.trainable_variables))
    return loss

def train(model, optimizer, t, x, u_real, u_imag, epochs):
    for epoch in range(epochs):
        loss = train_step(model, optimizer, t, x, u_real, u_imag)
        if epoch % 100 == 0:
            print(f'Epoch {epoch}, Loss: {loss.numpy()}')

def main():
    # Hyperparameters
    layers = [2, 20, 20, 2]
    learning_rate = 0.001
    epochs = 1000

    # Initialize the model and the optimizer
    model = ComplexPINN(layers)
    optimizer = tf.keras.optimizers.Adam(learning_rate=learning_rate)

    # Generate synthetic data
    N = 1000
    t = np.linspace(0, 1, N)
    x = np.linspace(-5, 5, N)
    t, x = np.meshgrid(t, x)
    u_real = np.exp(-0.5 * t) * np.cos(x)
    u_imag = np.exp(-0.5 * t) * np.sin(x)

    # Convert to TensorFlow tensors
    t = tf.convert_to_tensor(t.flatten()[:, None], dtype=tf.float32)
    x = tf.convert_to_tensor(x.flatten()[:, None], dtype=tf.float32)
    u_real = tf.convert_to_tensor(u_real.flatten()[:, None], dtype=tf.float32)
    u_imag = tf.convert_to_tensor(u_imag.flatten()[:, None], dtype=tf.float32)

    # Train the model
    train(model, optimizer, t, x, u_real, u_imag, epochs)

    # Generate test data
    t_test = tf.convert_to_tensor(np.random.uniform(0, 1, 100)[:, None], dtype=tf.float32)
    x_test = tf.convert_to_tensor(np.random.uniform(-5, 5, 100)[:, None], dtype=tf.float32)

    # Test the model
    u_test = model(tf.concat([t_test, x_test], 1))
    print('Test results:', u_test)

if __name__ == '__main__':
    main()
