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
- **核心功能**：基于pytorch实现各类推荐算法(DeepFM, WideDeep, DCN, DIN, DIEN)
- **主要区别**：相对于deep-ctr, 去除对tensorflow和keras的依赖，去除重复过embedding的操作，原生支持multiclass
- **训练过程**：
    ```text
    2022-10-28 23:16:10 - Start Training
    2022-10-28 23:16:10 - Epoch: 1/5
    5000/5000 [==============================] - 13s 3ms/step - loss: 0.1351 - acc: 0.9601
    Evaluate: 100%|██████████████████████████████████████████████████| 2500/2500 [00:03<00:00, 798.09it/s] 
    test_acc: 0.98045. best_test_acc: 0.98045

    2022-10-28 23:16:27 - Epoch: 2/5
    5000/5000 [==============================] - 13s 3ms/step - loss: 0.0465 - acc: 0.9862
    Evaluate: 100%|██████████████████████████████████████████████████| 2500/2500 [00:03<00:00, 635.78it/s] 
    test_acc: 0.98280. best_test_acc: 0.98280

    2022-10-28 23:16:44 - Epoch: 3/5
    5000/5000 [==============================] - 15s 3ms/step - loss: 0.0284 - acc: 0.9915
    Evaluate: 100%|██████████████████████████████████████████████████| 2500/2500 [00:03<00:00, 673.60it/s] 
    test_acc: 0.98365. best_test_acc: 0.98365

    2022-10-28 23:17:03 - Epoch: 4/5
    5000/5000 [==============================] - 15s 3ms/step - loss: 0.0179 - acc: 0.9948
    Evaluate: 100%|██████████████████████████████████████████████████| 2500/2500 [00:03<00:00, 692.34it/s] 
    test_acc: 0.98265. best_test_acc: 0.98365

    2022-10-28 23:17:21 - Epoch: 5/5
    5000/5000 [==============================] - 14s 3ms/step - loss: 0.0129 - acc: 0.9958
    Evaluate: 100%|██████████████████████████████████████████████████| 2500/2500 [00:03<00:00, 701.77it/s] 
    test_acc: 0.98585. best_test_acc: 0.98585

    2022-10-28 23:17:37 - Finish Training
    ```


## 3. 快速上手
- 参考了[deepctr-torch](https://github.com/shenweichen/DeepCTR-Torch), 使用[torch4keras](https://github.com/Tongjilibo/torch4keras)中作为Trainer
- [测试用例](https://github.com/Tongjilibo/rec4torch/tree/master/examples)


## 4. 版本说明
- **v0.0.2**：20240204 更新依赖项torch4keras版本
- **v0.0.1**：20221027 dcn, deepcrossing, deepfm, din, dien, wide&deep, ncf等模型，训练过程修改为传入dataloader，合并models和layers，合并简化embedding_lookup，去掉一些重复的embedding过程(提速)


## 5. 更新：
- **20240204**：更新依赖项torch4keras版本
- **20221110**：增加自定义的TensorDataset和collate_fn_device，支持指定device，防止显存占用多大，用out_dim和loss来替代task参数，兼容多分类
- **20221027**：增加deepcrossing, ncf, din, dien算法，使用torch4keras作为trainer
- **20220930**：初版提交, 训练过程修改为传入dataloader(参考bert4torch)，合并models和layers(模型结构较简单)，合并简化embedding_lookup，去掉一些重复的embedding过程(提速)