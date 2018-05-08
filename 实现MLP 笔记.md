# 实现MLP 笔记

将权重初始化维截断的正态分布，其标准差为0.1

```python
w1 = tf.Variable(tf.truncated_normal([in_units, h1_units], stddev=0.1))
```

之所以加噪声（0.1）是打破完全对称并且避免0梯度

## ReLU

$$
f(x) = \max(0,x)
$$

![](http://7pn4yt.com1.z0.glb.clouddn.com/blog-relu-perf.png)

