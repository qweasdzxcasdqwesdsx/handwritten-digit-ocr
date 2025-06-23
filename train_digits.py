#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""train_digits.py — 手写数字微调包装脚本（全中文注释）

⚠  背景说明
-------------------------------------------------------------------------------
ClovaAI 的 *deep-text-recognition-benchmark*（DTRB）仓库内置了一个通用
`train.py` 脚本，用于训练任意字符集（印刷体、自然场景文字……）。

在只识别 **数字 0‑9** 的场景下，我们不希望：
1. 去改动官方 `train.py` → 便于后续拉取更新。
2. 在命令行手动填写那一大堆与数字任务无关的超参数。

因此本文件充当“**薄包装器（wrapper）**”
- **解析**：仅暴露关键信息给终端用户（训练集、字符集、网络结构等）。
- **注入**：自动补齐 `train.py` 期望但用户不关心的其余字段（见 `_DEFAULTS`）。
- **预处理**：可在加载旧检查点时，删除原预测头，以便重新学习仅含数字的输出层。
- **委托**：最终调用动态加载的 `train.py::train()` 进入真正的训练循环。

用法示例
-------------------------------------------------------------------------------
```bash
python train_digits.py \
  --train_data data_digits_lmdb/train \
  --valid_data data_digits_lmdb/val \
  --character 0123456789 \
  --Transformation TPS --FeatureExtraction ResNet \
  --SequenceModeling BiLSTM --Prediction Attn
```
运行后会自动在 `saved_models/<exp_name>` 生成日志与权重。
"""

from __future__ import annotations  # 未来注解语法（Python 3.7+）

import argparse            # 解析命令行参数
import importlib.util      # 动态导入 train.py
import pathlib             # 面向对象的文件路径
import tempfile            # 生成临时文件，保存处理后的权重
from types import ModuleType

import torch               # PyTorch 主模块

# -----------------------------------------------------------------------------
# 1. 动态导入同目录下的官方 *train.py*
#    这样保持官方脚本“零修改”，方便随时 git pull 更新。
# -----------------------------------------------------------------------------

def _load_train_fn() -> callable:
    """把 `train.py` 当作模块临时加载，返回其 `train` 函数。"""
    spec = importlib.util.spec_from_file_location(
        "train_local", pathlib.Path(__file__).with_name("train.py")
    )
    train_local: ModuleType = importlib.util.module_from_spec(spec)  # type: ignore[arg-type]
    assert spec.loader is not None  # 静态类型检查
    spec.loader.exec_module(train_local)
    return train_local.train


train_fn = _load_train_fn()  # 以后直接用 train_fn 调用官方训练逻辑

# -----------------------------------------------------------------------------
# 2. 命令行接口（CLI）
#    只暴露“对数字识别重要”的参数，其他参数自动走默认。
# -----------------------------------------------------------------------------

def _get_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        description="在手写数字 (0‑9) 数据集上微调 ClovaAI TRBA 模型"
    )

    # ── 数据集路径 ────────────────────────────────────────────────
    p.add_argument("--train_data", required=True, help="训练集的 LMDB 文件夹")
    p.add_argument("--valid_data", required=True, help="验证集的 LMDB 文件夹")

    # ── 网络结构（与官方 train.py 保持一致的关键开关） ─────────────
    p.add_argument("--Transformation", required=True, choices=["TPS", "None"])
    p.add_argument("--FeatureExtraction", required=True, choices=["VGG", "RCNN", "ResNet"])
    p.add_argument("--SequenceModeling", required=True, choices=["BiLSTM", "None"])
    p.add_argument("--Prediction", required=True, choices=["CTC", "Attn"])

    # ── 字符集 & 图像尺寸 ────────────────────────────────────────
    p.add_argument("--character", required=True, help="任务中的全部字符 — 仅数字则填 0123456789")
    p.add_argument("--batch_max_length", type=int, default=1, help="单样本最大标签长度")
    p.add_argument("--imgH", type=int, default=32, help="输入图像高度")
    p.add_argument("--imgW", type=int, default=100, help="输入图像宽度")

    # ── 训练超参数（保持简洁，如果需要高阶功能可进官方 train.py） ─────
    p.add_argument("--lr", type=float, default=1e-4, help="学习率 (Adam / Adadelta)")
    p.add_argument("--num_iter", type=int, default=20000, help="训练迭代次数")
    p.add_argument("--batch_size", type=int, default=64, help="批大小")

    # ── 实验管理 ────────────────────────────────────────────────
    p.add_argument("--exp_name", default="digits_TRBA_finetune", help="子目录名")
    p.add_argument("--saved_model", default="", help="(可选) 预加载的权重路径")

    return p


# -----------------------------------------------------------------------------
# 3. 上游 *train.py* 期望但 CLI 未暴露的参数 → 这里给出安全默认值
# -----------------------------------------------------------------------------

_DEFAULTS: dict[str, object] = {
    # —— 通用设置 ——
    "data_filtering_off": False,   # 过滤无效样本
    "sensitive": False,            # 是否区分大小写
    "rgb": False,                  # 默认灰度图
    "PAD": True,                   # 保比例填充
    "manualSeed": 1111,
    "workers": 4,

    # —— 优化器相关 ——
    "adam": True,
    "beta1": 0.9,
    "rho": 0.95,
    "eps": 1e-8,
    "grad_clip": 5.0,
    "valInterval": 1000,

    # —— 模型结构 ——
    "num_fiducial": 20,
    "output_channel": 512,
    "hidden_size": 256,
    "FT": True,  # 是否微调 (Fine‑Tune)

    # —— 数据采样 ——
    "select_data": "train",       # 训练集子文件夹名
    "batch_ratio": "1.0",        # 单域，不需要平衡
    "total_data_usage_ratio": "1.0",

    # —— 其他 ——
    "baiduCTC": False,
}

# -----------------------------------------------------------------------------
# 4. 主入口
# -----------------------------------------------------------------------------

def main() -> None:
    # ① 解析命令行
    opt = _get_parser().parse_args()

    # ② 如果提供了检查点，先做“预测头裁剪”
    if opt.saved_model:
        ckpt = torch.load(opt.saved_model, map_location="cpu")
        # 删除旧字符集的预测层 → 重新学习 0‑9 的输出层
        for k in list(ckpt.keys()):
            if "Prediction" in k:
                del ckpt[k]
        # 保存到临时文件，再让 opt.saved_model 指向它
        tmp_pth = tempfile.NamedTemporaryFile(suffix=".pth", delete=False, dir=".").name
        torch.save(ckpt, tmp_pth)
        opt.saved_model = tmp_pth
        _DEFAULTS["FT"] = True  # 告诉 train.py “这是微调”
    else:
        _DEFAULTS["FT"] = False  # 从头训练

    # ③ 注入默认字段（官方 train.py 需要但 CLI 未提供的）
    for k, v in _DEFAULTS.items():
        if not hasattr(opt, k):
            setattr(opt, k, v)

    # ④ 创建实验目录
    log_dir = pathlib.Path("saved_models") / opt.exp_name
    log_dir.mkdir(parents=True, exist_ok=True)
    print(f"✔ 日志和检查点 >> {log_dir.resolve()}")
    print("✔ 字符集       >>", opt.character)
    print("✔ 批次最大长度  >>", opt.batch_max_length)

    # ⑤ 调用官方训练循环
    train_fn(opt)


if __name__ == "__main__":  # pragma: no cover
    main()
