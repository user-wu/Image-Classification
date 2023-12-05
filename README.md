# 图像分类
通用图像分类步骤，Pytorch实现。

## 数据准备
准备方式：
* 现有数据集
* 网络爬取数据

现有数据集不多介绍，主要介绍网络数据集爬取（因为我们需要的类别大概率不是现有数据集中的）；

数据爬取代码：1_claw.py，作用是爬取数据并生成数据集；
路径示例：

```
Dataset --------------------------
        |-cat         ｜-1.jpg
        |             ｜-2.jpg
        |             ｜-……
        --------------------------
        |-dog         ｜-1.jpg
        |             ｜-2.jpg
        |             ｜-3.jpg
        --------------------------
```
