import tensorflow as tf
from tensorflow.keras import layers, models, optimizers

class QNetwork(tf.keras.Model):
    def __init__(self, lr=0.01):
        super(QNetwork, self).__init__()
        # Input features: Nfloor, Npeople, Nstop, action/el.id
        self.dense1 = layers.Dense(8, activation='relu')
        self.dense2 = layers.Dense(1)
        
        # Build the model eagerly to instantiate variables before tf.function trace
        self.build(tf.TensorShape([None, 4]))
        
        # Optimizer and Loss
        self.optimizer = optimizers.Adam(learning_rate=lr)
        self.loss_fn = tf.keras.losses.MeanSquaredError()

    @tf.function
    def call(self, x):
        """
        x is a tensor of shape (batch, 4)
        Returns the Q-value Q(s, a).
        For cost minimization, lower Q is better.
        """
        out = self.dense1(x)
        out = self.dense2(out)
        return out

    @tf.function
    def update(self, s, target_q):
        """
        Perform a single gradient descent step.
        """
        with tf.GradientTape() as tape:
            pred_q = self(s)
            loss = self.loss_fn(target_q, pred_q)
            
        grads = tape.gradient(loss, self.trainable_variables)
        self.optimizer.apply_gradients(zip(grads, self.trainable_variables))
        
        return loss
