# 📣Introduction

一个非常easy的项目，仅仅只是为了通过猫狗分类实例熟悉基本的Tensorflow接口🎇

实验部分探索了三组不同的学习率对模型泛化性能的影响。

Furthermore，等之后兴致来潮就updating🤓~

# 💊Dependence

* python 3.8
* tensorflow 2.2.0
* tensorboardX 2.6
* Numpy 1.20.1
* matplotlib 3.7.1

# ✨Usage

### Dataset

下载地址：https://www.kaggle.com/datasets/tongpython/cat-and-dog

训练集、验证集、测试集比例：8 : 1 : 1

目录结构：

```bash
data/
├── test
│   ├── cats
│   │   ├── cat.1.jpg
│   │   ├── cat.2.jpg
│   │   ├── .
│   │   ├── .
│   └── dogs
├── train
│   ├── cats
│   └── dogs
└── val
    ├── cats
    └── dogs
```

### Checkpoints

链接：https://pan.baidu.com/s/13Sw-E-yE2xhlpP3TYwuTNg?pwd=3526 
提取码：3526



# 🎉Result

```bash
Training Accuracy:95.59%
Validation Accuracy:84.40% 
Testing Accuracy:83.58%
```



![r](F:\A-Curriculum\cv\lab\cat_dog_classification\figures\r.png)



# 👁‍🗨Reference

https://github.com/girishkuniyal/Cat-Dog-CNN-Classifier