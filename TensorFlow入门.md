# TensorFlow入门

## 人门用法
1.首先载入Tensorflow库,并创建InteractiveSession,接下来创建Placeholder,即输入数据的地方
```python
import Tensorflow as tf
session = tf.InteactiveSession()
x=tf.placeholder(tf.float32,[0,784]) # 第一个参数数据类型，第二个参数数据大小（0代表不限条数输入）
```
2.然后定义系数W and b
```python
w=tf.Variable(tf.zeros[784,10])
b=tf.Variable(tf.zeros(10))
```
3.实现算法（以softmax为例）
```python
y=tf.nn.softmax(tf.matmul(x,w)+b)
```
4.定义损失函数（loss Function)描述模型对分类的精度
常用的损失函数为交叉熵（cross-entropy),表示模型对真实概率的分布估计的准确程度
公式：
$$ H_{y'}(y)=-\sum y'_ilog(y_i)$$
```python
y_ = tf.placeholder(tf.float32,[None,10]
c = tf.reduce_mean(-tf.reduce_mean(y_*tf.log(y),
                                              reduction_indices=[1])) #reduction_indices=1表示每行运算 
```
5.定义优化算法
```python
train_step = tf.train.GradientDescentOptimizer(0.5).minimize(reduction_indices)
# 0.5为学习速率 reduction_indices为优化算法
```
6.使用全局参数初始化器
```python
tf.global_variables_initializer().run()
```
7.开始迭代的执行训练操作train_step,使用随机梯度下降，每次从训练集中挑出100个样本，构成一个mini_batch,并feed给Placeholder，然后调用train_step对样本进行训练。
```python
for i in range(1000):
    batch_xs,batch_ys = mnist.train.next_batch(100)
    train_step.run({x:batch_xs,y:batch_ys})
```
8.准确率验证
```python
correct_prediction = tf.equal(tf.argmax(y,1), tf.argmax(y_, 1))
accuracy = tf.reduce_mean(tf.cast(correct_prediction,tf.float32))
print(accuracy.eval({x:mnist.test.images,y_:mnist.test.labels}))
```
使用TensorFlow的四个部分：
1. 定义算法公式
2. 定义loss,选定优化器，并指定优化器去优化loss
3. 迭代的对数据进行训练
4. 在测试集或验证集对准确率进行评测


