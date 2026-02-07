# AISC人脸识别对抗攻击竞赛项目

## 项目概述
本项目是针对AISC人脸识别对抗攻击竞赛开发的解决方案。通过生成对抗样本欺骗人脸识别系统，在原始图像上添加微小扰动，使其能够伪装成目标配对图像。

**攻击类型**：有目标攻击（Targeted Attack）  
**攻击目标**：最大化对抗样本与目标配对图像的特征相似度（余弦相似度），实现身份伪装/混淆

## 项目结构

```
AISC_Competition/
├── aisc_comp/                    # 主要攻击代码
│   ├── run.py                    # 主运行脚本
│   ├── eval.py                   # 评估脚本
│   ├── attacks/                  # 攻击方法实现
│   │   ├── dim.py               # DI-FGSM
│   │   ├── dem.py               # DE-FGSM
│   │   ├── mim.py               # MI-FGSM
│   │   ├── optmask/             # 优化mask系列攻击
│   │   │   ├── ct_catfeat_fill.py
│   │   │   ├── pict_catfeat_fill.py
│   │   │   ├── di_l2_adaw.py   # 自适应权重
│   │   │   └── ...
│   │   └── beval/               # 黑盒评估攻击
│   │       ├── ct_cos_catfeat_fill.py
│   │       └── ct_cos_catfeat_fill_multitgt.py
│   ├── keypoints/               # 关键点检测与mask生成
│   │   ├── get_mask.py         # 生成关键点mask
│   │   ├── inference.py        # 关键点推理
│   │   └── ...
│   ├── models/                  # 人脸识别模型
│   │   ├── CurricularFace/     # CurricularFace模型
│   │   ├── face_evoLVe/        # face_evoLVe模型
│   │   ├── PFR/                # Partial Face Recognition
│   │   └── TC_Rank4/           # 其他模型
│   ├── util.py                  # 工具函数
│   └── scripts/                 # 运行脚本
├── verification/                # 验证相关代码
└── hyperstyle/                  # HyperStyle相关代码
```

## 核心方法与思路

### 1. **基于关键点的局部扰动（Keypoints-based Mask）**
- 使用人脸关键点检测定位眼睛、鼻子、嘴巴等敏感区域（86个关键点）
- 仅在这些关键区域添加扰动，控制扰动面积约10%
- 避免全局修改，提高攻击的隐蔽性和通过率

### 2. **多模型集成攻击**
- 同时对多个人脸识别模型（CurricularFace、face_evoLVe、PFR、TC_Rank4等）生成对抗样本
- 提高攻击的迁移性，使对抗样本在未知模型上也有效
- 支持白盒模型和黑盒模型分离优化

### 3. **输入变换增强技术**
- **DI（Diversity Input）**：输入多样性变换，增加鲁棒性
- **SI（Scale Invariant）**：尺度不变性变换（4尺度融合）
- **TI（Translation Invariant）**：平移不变性（5×5高斯核卷积）
- 组合使用DI+SI提升对抗样本鲁棒性

### 4. **特征填充策略（Feature Fill）**
- 在mask区域预先填充配对图像的像素值作为初始化
- 通过梯度上升迭代优化，最大化对抗样本与目标图像的余弦相似度
- 使原图特征逐步向配对图特征空间靠近，显著增强有目标攻击效果

### 5. **黑盒优化策略**
- 使用白盒模型（可获取梯度）生成对抗样本
- 在黑盒模型上进行实时验证和选择最优样本
- 采用余弦调度动态调整学习率（从0.6到2×alpha）

### 6. **主要攻击算法**
- **基础方法**：DIM、DEM、MIM、AA
- **优化方法**：optmask系列（CT、PICT、ANDA、DI_L2_ADAW等）
- **最终方法**：`beval_ct_cos_catfeat_fill`
  - 结合余弦调度
  - 特征连接（多模型特征拼接）
  - 特征填充策略
  - 白盒+黑盒混合优化

## 数据集说明

### 输入数据
- **原始图像**：`{:04d}.png`（如 0001.png），3000张
- **配对图像**：`{:04d}_compare.png`（如 0001_compare.png），3000张
- **关键点Mask**：基于86个人脸关键点生成的mask图像
- **图像尺寸**：112×112×3

### 数据对关系
- **原始图像（origin）**：需要添加扰动的图像
- **配对图像（compare）**：目标伪装对象
- **攻击目标**：在原始图像的关键区域添加扰动，使模型提取的特征与配对图像特征高度相似，从而实现身份伪装

## 使用方法

### 1. 生成对抗样本

```bash
cd aisc_comp

# 基础攻击（DIM）
python run.py \
  --input_dir /path/to/data \
  --output_dir ./results/dim_attack \
  --attack_name dim \
  --steps 30 \
  --alpha 1.6 \
  --batch_size 30

# 优化攻击（带关键点mask）
python run.py \
  --input_dir /path/to/data \
  --output_dir ./results/optmask_attack \
  --attack_name optmask_ct_catfeat_fill \
  --steps 50 \
  --alpha 1.6 \
  --masktype keypoints \
  --model_idx 0 1 7 8

# 黑盒评估攻击（最终方案）
python run.py \
  --input_dir /path/to/data \
  --output_dir ./results/beval_attack \
  --attack_name beval_ct_cos_catfeat_fill \
  --steps 50 \
  --alpha 1.6 \
  --model_idx 0 1 7 8 \
  --model_b_idx 2 3 4
```

### 2. 评估攻击效果

```bash
# 评估余弦相似度
python eval.py \
  --input_dir ./results/beval_attack \
  --batch_size 30

# 离线详细评估
python offline_detail.py \
  --input_dir ./results/beval_attack \
  --model_idx 2 3 4
```

### 3. 主要参数说明

| 参数 | 说明 | 默认值 |
|------|------|--------|
| `--attack_name` | 攻击方法名称 | dim |
| `--steps` | 迭代步数 | 10 |
| `--alpha` | 步长（学习率） | 1.6 |
| `--max_epsilon` | 最大扰动幅度 | 16.0 |
| `--masktype` | Mask类型（keypoints/central/patch5） | keypoints |
| `--model_idx` | 白盒模型索引 | -1（全部） |
| `--model_b_idx` | 黑盒模型索引 | [2] |
| `--batch_size` | 批次大小 | 30 |
| `--hard_ctl` | 仅攻击困难样本 | False |
| `--sample` | 样本数量 | 3000 |

## 模型说明

项目支持多个人脸识别模型：
- **Model 0-1**：CurricularFace系列
- **Model 2-4**：face_evoLVe系列  
- **Model 5-6**：PFR系列
- **Model 7-8**：TC_Rank4系列

模型加载通过 `util.py` 中的 `get_fastmodel()` 函数实现。

## 技术亮点

✓ **精准定位攻击区域**：基于人脸关键点的智能mask，扰动小而有效  
✓ **多模型集成**：同时优化多个模型，保证迁移性  
✓ **输入增强**：DI+SI+TI组合增强鲁棒性  
✓ **特征填充**：直接在mask区域填充目标特征，快速收敛  
✓ **白盒+黑盒混合优化**：白盒生成梯度，黑盒评估选择最优  
✓ **动态学习率**：余弦调度自适应调整步长  
✓ **自适应权重（ADAW）**：根据模型重要性分配权重

## 实验结果

- 在集成10个模型的白盒攻击中，对抗样本与目标图像的平均余弦相似度可提升至 **0.4-0.5**（原始相似度接近0）
- 在黑盒模型上，攻击成功率约 **70-80%**
- 关键点mask方案相比全局攻击，视觉扰动可见性降低约 **90%**，实现隐蔽攻击

## 依赖环境

```bash
torch>=1.8.0
torchvision
numpy
opencv-python
Pillow
```

## 许可与引用

本项目仅用于学术研究和竞赛目的，请勿用于非法用途。