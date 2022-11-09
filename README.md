# rec4torch
推荐系统的pytorch算法实现

[![licence](https://img.shields.io/github/license/Tongjilibo/rec4torch.svg?maxAge=3600)](https://github.com/Tongjilibo/rec4torch/blob/master/LICENSE) 
[![PyPI](https://img.shields.io/pypi/v/rec4torch?label=pypi%20package)](https://pypi.org/project/rec4torch/) 
[![PyPI - Downloads](https://img.shields.io/pypi/dm/rec4torch)](https://pypistats.org/packages/rec4torch)
[![GitHub stars](https://img.shields.io/github/stars/Tongjilibo/rec4torch?style=social)](https://github.com/Tongjilibo/rec4torch)
[![GitHub Issues](https://img.shields.io/github/issues/Tongjilibo/rec4torch.svg)](https://github.com/Tongjilibo/rec4torch/issues)
[![contributions welcome](https://img.shields.io/badge/contributions-welcome-brightgreen.svg?style=flat)](https://github.com/Tongjilibo/rec4torch/issues)

## 1. 下载安装
安装稳定版
```shell
pip install rec4torch
```
安装最新版
```shell
pip install git+https://www.github.com/Tongjilibo/rec4torch.git
```


## 2. 功能
- 基于pytorch实现各类推荐算法(DeepFM, WideDeep, DCN, DIN, DIEN)


## 3. 快速上手
- 参考了[deepctr-torch](https://github.com/shenweichen/DeepCTR-Torch), 使用[torch4keras](https://github.com/Tongjilibo/torch4keras)中作为Trainer
- [测试用例](https://github.com/Tongjilibo/rec4torch/tree/master/examples)


## 4. 版本说明
- **v0.0.1**：20221027 dcn, deepcrossing, deepfm, din, dien, wide&deep, ncf等模型，训练过程修改为传入dataloader，合并models和layers，合并简化embedding_lookup，去掉一些重复的embedding过程(提速)


## 5. 更新：
- **20221027**：增加deepcrossing, ncf, din, dien算法，使用torch4keras作为trainer
- **20220930**：初版提交, 训练过程修改为传入dataloader(参考bert4torch)，合并models和layers(模型结构较简单)，合并简化embedding_lookup，去掉一些重复的embedding过程(提速)