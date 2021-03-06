﻿# numpy

标签（空格分隔）： numpy 数据挖掘 

---
## 切片
数组切片是原始数组的视图。这意味着数据不会被复制，视图上的任何修改都会直接反映到源数组上
```python
In [16]: arr
Out[16]: array([1, 2, 3, 4, 5, 6, 7, 8, 9])

In [17]: arr_slice = arr[3:6]

In [18]: arr_slice[:]=5

In [19]: arr_slice
Out[19]: array([5, 5, 5])

In [20]: arr
Out[20]: array([1, 2, 3, 5, 5, 5, 7, 8, 9])

```
## 布尔型索引
通过布尔型索引选取数组中的数据，总会产生创建数据的副本，即使返回一模一样的数组也是如此。
```python
In [23]: arr = np.empty((8,4))

In [24]: for i in range(8):
    ...:     arr[i] = i
In [25]: arr[[4,3,2]]
Out[25]:
array([[ 4.,  4.,  4.,  4.],
       [ 3.,  3.,  3.,  3.],
       [ 2.,  2.,  2.,  2.]])
       
       
In [27]: arr = np.arange(32).reshape((8,4))
In [28]: arr
Out[28]:
array([[ 0,  1,  2,  3],
       [ 4,  5,  6,  7],
       [ 8,  9, 10, 11],
       [12, 13, 14, 15],
       [16, 17, 18, 19],
       [20, 21, 22, 23],
       [24, 25, 26, 27],
       [28, 29, 30, 31]])
In [29]: arr[[1,5,7,2],[0,3,1,2]]
Out[29]: array([ 4, 23, 29, 10])
In [30]: arr[[1,5,7,2]][:,[0,3,1,2]]
Out[30]:
array([[ 4,  7,  5,  6],
       [20, 23, 21, 22],
       [28, 31, 29, 30],
       [ 8, 11,  9, 10]])
```



