## 快速开始
## 模型下载
- 百度网盘
- 通过网盘分享的文件：model
链接: https://pan.baidu.com/s/1IL1LVStUZ4Mw_tTNb4E3aQ?pwd=1234 提取码: 1234 
--来自百度网盘超级会员v4的分享
### 依赖
- 实验环境：**PyTorch 2.3.0 + CUDA 12.1 + Python 3.10 + Ubuntu 22.04**

  ```bash
  # 官方提供的 cu121 轮子
  pip3 install torch==2.3.0+cu121 torchvision==0.18.0+cu121 \
    --extra-index-url https://download.pytorch.org/whl/cu121
如需复现论文的老版结果，可回退至 PyTorch 0.4.1 + CUDA 9.0。
其他依赖
pip3 install lmdb pillow torchvision nltk natsort
## 训练集选择
- 北京大学开源SVHN
- MNIST + EMNIST 可自行根据tensorflow或pytorch下载使用
# 1. 训练 CRNN
CUDA_VISIBLE_DEVICES=0 python3 train.py \
  --train_data data_lmdb_release/training \
  --valid_data data_lmdb_release/validation \
  --select_data MJ-ST --batch_ratio 0.5-0.5 \
  --Transformation None --FeatureExtraction VGG \
  --SequenceModeling BiLSTM --Prediction CTC
# 2. 测试 CRNN
CUDA_VISIBLE_DEVICES=0 python3 test.py \
  --eval_data data_lmdb_release/evaluation --benchmark_all_eval \
  --Transformation None --FeatureExtraction VGG \
  --SequenceModeling BiLSTM --Prediction CTC \
  --saved_model saved_models/None-VGG-BiLSTM-CTC-Seed1111/best_accuracy.pth

# 3. 训练 / 测试最佳模型 TRBA
CUDA_VISIBLE_DEVICES=0 python3 train.py \
  --train_data data_lmdb_release/training \
  --valid_data data_lmdb_release/validation \
  --select_data MJ-ST --batch_ratio 0.5-0.5 \
  --Transformation TPS --FeatureExtraction ResNet \
  --SequenceModeling BiLSTM --Prediction Attn
常用参数
参数	说明
--train_data / --valid_data / --eval_data	lmdb 数据路径
--select_data	选择训练集（默认 MJ-ST）
--batch_ratio	每个训练集在 batch 中的比例
--Transformation	图像变换 `[None
--FeatureExtraction	特征提取 `[VGG
--SequenceModeling	序列建模 `[None
--Prediction	预测 `[CTC
--saved_model	指定模型进行评估
--benchmark_all_eval	在 10 个评测集版本上评估

引用
@inproceedings{baek2019STRcomparisons,
  title={What Is Wrong With Scene Text Recognition Model Comparisons? Dataset and Model Analysis},
  author={Baek, Jeonghun et al.},
  booktitle={ICCV},
  year={2019}
}

代码 / 论文：Jeonghun Baek ku21fang@gmail.com
