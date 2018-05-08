# IPython的使用

标签（空格分隔）： IPython 

---

## 内省
变量前面或后面加?显示有关该对象的一些通用信息
```python
In [3]: a = ([1,2,3,4])
In [4]: a?
Type:        list
String form: [1, 2, 3, 4]
Length:      4
Docstring:
list() -> new empty list
list(iterable) -> new list initialized from iterable's items

In [7]: np.*load*?
np.__loader__
np.load
np.loads
np.loadtxt
np.pkgload
```
## %run命令
%paste 和 %cpaste可以承接剪切板中的一切文本，并在shell中以整体执行
## %time和%timeit
% time一次执行一条语句，然后报告总体的执行时间
% timeit多次运行以产生一个非常精确的平均执行时间





