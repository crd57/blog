# Tensorflow(自编码器）
**神经网络的最大价值在于对特征的自动提取和抽象，可以找出复杂其有效的高阶特征**
##稀疏表达
是用少量的基本特征组合拼装形成更高阶层的抽象特征
##自编码器
使用本身的高阶特征来编码自己。
他的输入输出一致，目标是使用高阶特征来组合自己
使用无监督的逐层训练的贪心算法。
Hinton提出的DBN模型包含多个隐含层，每个隐藏层都是限制性玻尔兹曼机RBM。
###去噪自编码器
参数初始化方法：xavier initialization,特点是根据某一层输入，输出节点数量自动调整最合适的分布
Xavier Glorot和Yoshua Bengio在一篇论文中指出，如果深度学习的权重初始化得太小，信号将在传播过程中逐渐缩小难以产生作用，如果权重初始化的太大，难么将在传播过程中逐步放大，导致弥散。Xavier的着用就是初始化的不大不小。
权重满足0均值，方差为$2/(n_{in}+n_{out})$,分布可以用均值分布或高斯分布。
```python
def xavier_init(fan_in,fan_out,constant=1):
    low = -constant * np.sqrt(6.0/(fan_in+fan_out))
    high = constant * np.sqrt(6.0/(fan_out+fan_in))
    return tf.random_uniform((fan_in, fan_out),
                             minval=low,maxval=high,
                             dtype = tf.float32)
```
范围在$-\sqrt{\frac{6}{n_{in}+n_{out}}},\sqrt{\frac{6}{n_{in}+n_{out}}}$的均与分布，$fan_{in}$为输入的节点数目，$fan_{out}$为输出的节点数目。
接下来定义一个去噪自编码的class
```python
class AdditiveGaussianNoiseAutoEncoder(object):
    def __init__(self,n_input,n_hidden,transfer_function=tf.nn.softplus,
                 optimizer = tf.train.AdamOptimizer(),scale=0.1):
        """

        :param n_input: 输入变量数
        :param n_hidden: 隐含层节点数
        :param transfer_function: 隐含层激活函数，默认为SoftPlus
        :param optimizer: 优化器，默认为Adam
        :param scale: 高斯噪声系数，默认为0.1
        """
        self.n_input = n_input
        self.n_hidden = n_hidden
        self.transfer = transfer_function
        self.scale = tf.placeholder(tf.float32)
        self.training_scale = scale
        network_weights = self._initialize_weights()
        self.weights = network_weights
        self.x = tf.placeholder(tf.float32,[None, self.n_input])
        self.hidden = self.transfer(tf.add(tf.matmul(self.x+scale*tf.random_normal((n_input,)), 
                                                     self.weights['w1']),self.weights['b1']))
                                                     # 给x加上噪声
                                                     # 加上噪声的x乘权重加上偏置
        self.reconstruction = tf.add(tf.matmul(self.hidden,
                                               self.weights['w2']),self.weights['b2'])
        self.cost = 0.5 * tf.reduce_sum(tf.pow(tf.subtract(
            self.reconstruction,self.x),2.0))   # 直接使用平方误差作为cost
        self.optimizer = optimizer.minimize(self.cost)
        # 定义训练操作为优化器optimizer对损失self.cost进行优化
        init= tf.global_variables_initializer() #全局变量初始化
        self.sess = tf.Session()
        self.sess.run(init)
        
```
参数初始化函数 _initialize_weights,先创建一个名为all_weights的字典，然后将w1,b1,w2,b2全部存入，最后返回all_weights.其中w1需要使用前面定义的xavier_init函数初始化。
```python
    def _initialize_weights(self):
        all_weights = dict()
        all_weights['w1'] = tf.Variable(xavier_init(self.n_input,
                                                    self.n_hidden))
        all_weights['b1'] = tf.Variable(tf.zeros([self.n_hidden],
                                                 dtype=tf.float32))
        all_weights['w2'] = tf.Variable(tf.zeros([self.n_hidden,
                                                  self.n_input],dtype=tf.float32))
        all_weights['b2'] = tf.Variable(tf.zeros([self.n_input],
                                                 dtype = tf.float32))
        return all_weights
```
定义计算损失cost以及执行一步训练的函数partial_fit.函数里只需让Session执行两个计算图的节点，分别是损失cost和训练过程optimizer，输入的feed_dict包括输入数据X，以及噪声的系数scale。函数partial——fit做的就是用一个batch数据进行训练并返回当前的损失cost
