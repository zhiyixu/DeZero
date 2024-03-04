# 为了解决类之间循环引用的问题，然后在这里定义基类作为顶级类
# 这个顶级类用来做变量声明
from abc import ABC


class BaseVariable(ABC):
    ...


class BaseFunction(ABC):
    ...
