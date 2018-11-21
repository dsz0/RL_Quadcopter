from keras import layers, models, optimizers
from keras import backend as K

#这里是评论者模型，比actor要简单。
class Critic:
    """Critic (Value) Model."""

    def __init__(self, state_size, action_size):
        """Initialize parameters and build model.

        Params
        ======
            state_size (int): Dimension of each state
            action_size (int): Dimension of each action
        """
        self.state_size = state_size
        self.action_size = action_size

        # Initialize any other variables here

        self.build_model()

    def build_model(self):
        """Build a critic (value) network that maps (state, action) pairs -> Q-values."""
        # Define input layers 定义输入层，将（状态，动作）对映射到Q值。
        states = layers.Input(shape=(self.state_size,), name='states')
        actions = layers.Input(shape=(self.action_size,), name='actions')

        # Add hidden layer(s) for state pathway 添加状态模型路径，和Actor保持一致。
        net_states = layers.Dense(units=32, activation='relu')(states)
        net_states = layers.Dense(units=64, activation='relu')(net_states)
        #net_states = layers.Dense(units=400,kernel_regularizer=layers.regularizers.l2(1e-6))(states)
        #net_states = layers.BatchNormalization()(net_states)
        #net_states = layers.Activation("relu")(net_states)
        #net_states = layers.Dense(units=300, kernel_regularizer=layers.regularizers.l2(1e-6))(net_states)

        # Add hidden layer(s) for action pathway
        net_actions = layers.Dense(units=32, activation='relu')(actions)
        net_actions = layers.Dense(units=64, activation='relu')(net_actions)

        #net_actions = layers.Dense(units=300,kernel_regularizer=layers.regularizers.l2(1e-6))(actions)
        #net_actions = layers.BatchNormalization()(net_actions)
        #net_actions = layers.Activation("relu")(net_actions)
        #net_actions = layers.Dense(units=300, kernel_regularizer=layers.regularizers.l2(1e-6))(net_actions)

        # Try different layer sizes, activations, add batch normalization, regularizers, etc.
        # 尝试在这里添加更多层级，改变layer size,添加激活层，batchnormalization层，

        # Combine state and action pathways 这两个层级先通过单独的路径（mini network）处理。
        # 下面将两个层级通过add结合到了一起。
        net = layers.Add()([net_states, net_actions])
        net = layers.Activation('relu')(net)

        # Add more layers to the combined network if needed

        # Add final output layer to prduce action values (Q values)
        Q_values = layers.Dense(units=1, name='q_values')(net)

        # Create Keras model
        self.model = models.Model(inputs=[states, actions], outputs=Q_values)

        # Define optimizer and compile model for training with built-in loss function
        optimizer = optimizers.Adam(lr=0.001)
        self.model.compile(optimizer=optimizer, loss='mse')

        # Compute action gradients (derivative of Q values w.r.t. to actions)
        #最终输出，任何给定（状态，动作）对的Q值。我们还需要计算此Q值对于相应动作向量的梯度。
        #梯度用于训练行动者模型。这一步需要单独执行，定义了get_action_gradients来访问这些梯度。
        action_gradients = K.gradients(Q_values, actions)

        # Define an additional function to fetch action gradients (to be used by actor model)
        self.get_action_gradients = K.function(
            inputs=[*self.model.input, K.learning_phase()],
            outputs=action_gradients)
        
        