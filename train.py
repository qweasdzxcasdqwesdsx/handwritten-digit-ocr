# -*- coding: utf-8 -*-
"""
train.py — 基于 deep-text-recognition-benchmark 的训练脚本

主要功能：
1. 解析命令行参数，配置训练与推理的超参数；
2. 准备训练、验证数据集（支持多源数据批次均衡采样 Batch Balanced Dataset）；
3. 构建文本识别模型（可按需组合 TPS、ResNet、BiLSTM、CTC / Attn 等模块）；
4. 进入训练循环，实时计算损失并定期在验证集评估，保存最佳模型；
5. 支持多 GPU、断点续训、详细日志与超参数保存。

"""

import os      # 操作系统相关，如路径拼接、文件读写
import sys     # 与 Python 解释器交互，可动态修改搜索路径
import time    # 计时工具
import random  # 随机数生成
import string  # 字符串常量与工具
import argparse  # 命令行参数解析

import torch  # PyTorch 主模块
import torch.backends.cudnn as cudnn  # cuDNN 后端配置
import torch.nn.init as init  # 参数初始化工具
import torch.optim as optim  # 优化器
import torch.utils.data  # 数据加载器
import numpy as np

# 项目自定义工具与模块\ nfrom utils import (
    CTCLabelConverter,
    CTCLabelConverterForBaiduWarpctc,
    AttnLabelConverter,
    Averager,
)
from dataset import hierarchical_dataset, AlignCollate, Batch_Balanced_Dataset
from model import Model
from test import validation

# 根据硬件情况选择设备
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def train(opt):
    """训练主函数

    参数
    ------
    opt : argparse.Namespace
        解析后的命令行参数集合。
    """

    """ 数据集准备 """
    if not opt.data_filtering_off:
        print("过滤掉标签包含非目标字符或长度超过 batch_max_length 的样本……")

    # 按 “-” 拆分数据源与批次比例
    opt.select_data = opt.select_data.split("-")
    opt.batch_ratio = opt.batch_ratio.split("-")

    # 构建带比例均衡的训练数据集
    train_dataset = Batch_Balanced_Dataset(opt)

    # 记录数据集信息
    log = open(f"./saved_models/{opt.exp_name}/log_dataset.txt", "a")

    # 构建验证集 DataLoader
    AlignCollate_valid = AlignCollate(imgH=opt.imgH, imgW=opt.imgW, keep_ratio_with_pad=opt.PAD)
    valid_dataset, valid_dataset_log = hierarchical_dataset(root=opt.valid_data, opt=opt)
    valid_loader = torch.utils.data.DataLoader(
        valid_dataset,
        batch_size=opt.batch_size,
        shuffle=not opt.valid_dataset_shuffle,
        num_workers=int(opt.workers),
        collate_fn=AlignCollate_valid,
        pin_memory=True,
    )
    log.write(valid_dataset_log)
    print("-" * 80)
    log.write("-" * 80 + "\n")
    log.close()

    """ 模型配置 """
    # 选择标签编码器（CTC / Attn）
    if "CTC" in opt.Prediction:
        converter = (
            CTCLabelConverterForBaiduWarpctc(opt.character)
            if opt.baiduCTC
            else CTCLabelConverter(opt.character)
        )
    else:
        converter = AttnLabelConverter(opt.character)
    opt.num_class = len(converter.character)

    # 根据是否使用彩色图像修改通道数
    if opt.rgb:
        opt.input_channel = 3

    # 实例化模型
    model = Model(opt)
    print(
        "模型输入参数：",
        opt.imgH,
        opt.imgW,
        opt.num_fiducial,
        opt.input_channel,
        opt.output_channel,
        opt.hidden_size,
        opt.num_class,
        opt.batch_max_length,
        opt.Transformation,
        opt.FeatureExtraction,
        opt.SequenceModeling,
        opt.Prediction,
    )

    # 参数初始化
    for name, param in model.named_parameters():
        if "localization_fc2" in name:
            print(f"跳过 {name}（TPS 本身已初始化）")
            continue
        try:
            if "bias" in name:
                init.constant_(param, 0.0)
            elif "weight" in name:
                init.kaiming_normal_(param)
        except Exception:  # batchnorm 等特殊层
            if "weight" in name:
                param.data.fill_(1)
            continue

    # 启用多 GPU
    model = torch.nn.DataParallel(model).to(device)

    # 损失均值器
    loss_avg = Averager()

    # 仅选择需要梯度的参数
    filtered_parameters, params_num = [], []
    for p in filter(lambda p: p.requires_grad, model.parameters()):
        filtered_parameters.append(p)
        params_num.append(np.prod(p.size()))
    print("可训练参数总数：", sum(params_num))

    # 优化器（Adam / Adadelta）
    if opt.adam:
        optimizer = optim.Adam(filtered_parameters, lr=opt.lr, betas=(opt.beta1, 0.999))
    else:
        optimizer = optim.Adadelta(filtered_parameters, lr=opt.lr, rho=opt.rho, eps=opt.eps)
    print("优化器配置：")
    print(optimizer)

    """ 保存完整超参数配置 """
    with open(f"./saved_models/{opt.exp_name}/opt.txt", "a") as opt_file:
        opt_log = "------------ 运行参数 -------------\n"
        for k, v in vars(opt).items():
            opt_log += f"{k}: {v}\n"
        opt_log += "---------------------------------\n"
        print(opt_log)
        opt_file.write(opt_log)

    """ 断点续训 """
    start_iter = 0
    if opt.saved_model:
        try:
            start_iter = int(opt.saved_model.split("_")[-1].split(".")[0])
            print(f"继续训练，起始迭代：{start_iter}")
        except ValueError:
            print("无法从模型文件名解析起始迭代，将从 0 开始。")
        print(f"加载预训练模型：{opt.saved_model}")
        model.load_state_dict(torch.load(opt.saved_model, map_location=device))

    # 训练集 DataLoader
    AlignCollate_train = AlignCollate(imgH=opt.imgH, imgW=opt.imgW, keep_ratio_with_pad=opt.PAD)
    train_loader = torch.utils.data.DataLoader(
        train_dataset,
        batch_size=opt.batch_size,
        shuffle=True,
        num_workers=int(opt.workers),
        collate_fn=AlignCollate_train,
        pin_memory=True,
    )

    # 损失函数
    if "CTC" in opt.Prediction:
        criterion = torch.nn.CTCLoss(zero_infinity=True).to(device)
    else:
        criterion = torch.nn.CrossEntropyLoss(ignore_index=0).to(device)
    print("损失函数：", criterion)

    # 初始化评估指标
    best_accuracy, best_norm_ED = -1, -1
    iteration = start_iter

    # 记录计时
    global start_time
    start_time = time.time()

    while True:  # 主训练循环
        # 取一个批次
        image_tensors, labels = train_dataset.get_batch()
        image = image_tensors.to(device)
        text, length = converter.encode(labels, batch_max_length=opt.batch_max_length)
        batch_size = image.size(0)

        # 前向传播与损失计算
        if "CTC" in opt.Prediction:
            preds = model(image, text)
            preds_size = torch.IntTensor([preds.size(1)] * batch_size)
            if opt.baiduCTC:
                preds = preds.permute(1, 0, 2)
            cost = criterion(preds.log_softmax(2), text, preds_size, length)
        else:
            preds = model(image, text[:, :-1])  # 去掉 [GO]
            target = text[:, 1:]  # 去掉 [GO]
            cost = criterion(preds.view(-1, preds.shape[-1]), target.contiguous().view(-1))

        # 反向传播
        model.zero_grad()
        cost.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), opt.grad_clip)
        optimizer.step()

        # 记录平均损失
        loss_avg.add(cost)

        # 周期性验证
        if (iteration + 1) % opt.valInterval == 0:
            elapsed_time = time.time() - start_time
            start_time = time.time()
            with torch.no_grad():
                model.eval()
                valid_loss, current_accuracy, current_norm_ED, preds_str, labels_str, conf_score = validation(
                    model, criterion, valid_loader, converter, opt
                )
                model.train()

            # 打印与记录日志
            loss_log = (
                f"[{iteration+1}/{opt.num_iter}] 训练损失: {loss_avg.val():.5f}, "
                f"验证损失: {valid_loss:.5f}, 用时: {elapsed_time:.2f}s"
            )
            loss_avg.reset()

            current_model_log = f"当前准确率: {current_accuracy:.3f}, 当前 Norm_ED: {current_norm_ED:.2f}"

            # 保存最佳模型
            if current_accuracy > best_accuracy:
                best_accuracy = current_accuracy
                torch.save(model.state_dict(), f"./saved_models/{opt.exp_name}/best_accuracy.pth")
            if current_norm_ED > best_norm_ED:
                best_norm_ED = current_norm_ED
                torch.save(model.state_dict(), f"./saved_models/{opt.exp_name}/best_norm_ED.pth")
            best_model_log = f"最佳准确率: {best_accuracy:.3f}, 最佳 Norm_ED: {best_norm_ED:.2f}"

            full_log = f"{loss_log}\n{current_model_log}\n{best_model_log}"
            print(full_log)
            with open(f"./saved_models/{opt.exp_name}/log_train.txt", "a") as log_f:
                log_f.write(full_log + "\n")

            # 打印部分预测结果
            dashed_line = "-" * 80
            head = f"{'真实标签':25s} | {'预测结果':25s} | 置信度 & 正确与否"
            result_log = f"{dashed_line}\n{head}\n{dashed_line}\n"
            for gt, pred, conf in zip(labels_str[:5], preds_str[:5], conf_score[:5]):
                if "Attn" in opt.Prediction:
                    gt = gt[: gt.find("[s]")]
                    pred = pred[: pred.find("[s]")]
                result_log += f"{gt:25s} | {pred:25s} | {conf:.4f}\t{pred == gt}\n"
            result_log += dashed_line
            print(result_log)
            with open(f"./saved_models/{opt.exp_name}/log_train.txt", "a") as log_f:
                log_f.write(result_log + "\n")

        # 每 1e5 迭代保存一次模型
        if (iteration + 1) % 1e5 == 0:
            torch.save(
                model.state_dict(), f"./saved_models/{opt.exp_name}/iter_{iteration + 1}.pth"
            )

        # 训练结束条件
        if (iteration + 1) == opt.num_iter:
            print("训练结束！")
            sys.exit()

        iteration += 1


if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    """ 训练相关超参数 """
    parser.add_argument("--exp_name", help="日志和模型保存目录", required=True)
    parser.add_argument("--train_data", required=True, help="训练集路径")
    parser.add_argument("--valid_data", required=True, help="验证集路径")
    parser.add_argument("--manualSeed", type=int, default=1111, help="随机种子")
    parser.add_argument("--workers", type=int, default=4, help="DataLoader 线程数")
    parser.add
