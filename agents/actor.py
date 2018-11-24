from keras import layers, models, optimizers
from keras import backend as K

#backend as K 用K作为一个通用的后端，算是约定成俗；但把K换成tensorflow我感觉会更明确 
#行动者（策略）模型
#这个会使用Keras定义的一个非常简单的行动者模型,这是一个包含数个隐藏层的
class Actor:
    """Actor (Policy) Model."""

    def __init__(self, state_size, action_size, action_low, action_high):
        """Initialize parameters and build model.
        Params
        ======
           state_size (int): Dimension of each state 状态的维度
           action_size (int): Dimension of each action 动作的维度
           action_low (array): Min value of each action dimension
           action_high (array): Max value of each action dimension
        """
        self.state_size = state_size
        self.action_size = action_size
        self.action_low = action_low
        self.action_high = action_high
        self.action_range = self.action_high - self.action_low

        # Initialize any other variables here 还能初始化其他需要的变量，不过暂时没什么想法。
        self.build_model()

    def build_model(self):
        """Build an actor (policy) network that maps states -> actions."""
        # Define input layer (states)建立策略模型，状态映射动作的模型。
        # 首先定义输入，也就是环境状态，这个状态的计量维度，由数字表现
        states = layers.Input(shape=(self.state_size,), name='states')
        
        # Add hidden layers        # 添加中间层，
        net = layers.Dense(units=256)(states)
        net = layers.BatchNormalization()(net)
        net = layers.Activation("relu")(net)
        
        net = layers.Dense(units=512)(net)
        net = layers.BatchNormalization()(net)
        net = layers.Activation("relu")(net)
        
        # Try different layer sizes, activations, add batch normalization, regularizers, etc.
        # 这里全部使用Dense 和 relu有点不太符合深度学习的建议，全连接层，提取特征和泛化
        # 在多层深度学习中添加BatchNormalization层，将前一层激活值重新规范化，
        #使得其输出数据 均值接近0，标准差接近1。可以控制过拟合，加速收敛
        # Add final output layer with sigmoid activation 最后的输出层要使用sigmoid激活函数activation='sigmoid', name='raw_actions')(net)
        raw_actions = layers.Dense(units=self.action_size, activation='sigmoid',
                                   name='raw_actions')(net)
        
        # Scale [0, 1] output for each action dimension to proper range
        # 输出层生成的原始动作，位于[0.0, 1.0]范围之内(因为使用了sigmoid激活函数)。
        # 添加层级，针对每个动作维度 将输出缩放到期望的范围。针对任何给定状态向量生成确定动作。
        # Lambda层的意思我还不是太清楚！这个函数用以对上一层的输出施以任何Theano/TensorFlow表达式。function：要实现的函数，该函数仅接受一个变量，即上一层的输出。
        actions = layers.Lambda(lambda x: (x * self.action_range) + self.action_low,
                                name='actions')(raw_actions)

        # Create Keras model
        self.model = models.Model(inputs=states, outputs=actions)
        # Define loss function using action value (Q value) gradients 定义损失函数loss, 使用了动作值的梯度.
        #shape 是一个尺寸元组(整数),不包含批量大小。 not including the batch size. shape=(32,) 表明期望的输入是按批次的32维向量。batch_shape 包含批量大小的元组。batch_shape=(10, 32) 表明期望的输入是 10 个 32 维向量。batch_shape=(None, 32) 表明任意批次大小的 32 维向量。
        action_gradients = layers.Input(shape=(self.action_size,))
        loss = K.mean(-action_gradients * actions)

        # Incorporate any additional losses here (e.g. from regularizers)

        # Define optimizer and training function 定义好优化器和训练函数。
        optimizer = optimizers.Adam()
        updates_op = optimizer.get_updates(params=self.model.trainable_weights, loss=loss)
        self.train_fn = K.function(
            inputs=[self.model.input, action_gradients, K.learning_phase()],
            outputs=[],
            updates=updates_op)