# ScienceDB（科学数据银行）上传信息（中文，直接复制粘贴）

> 网站：https://www.scidb.cn   （本机实测可正常访问；免费发布，提供 DOI + CSTR 双标识）
> 上传文件：`S2-LithoBench_v1.0_code_and_data.zip`（5.8 MB）

## 数据集名称

S2-LithoBench v1.0：面向验证可信的 Sentinel-2 岩性填图基准的代码与衍生数据

## 英文名称（如页面要求）

S2-LithoBench v1.0: code and derived data for a validation-aware Sentinel-2
lithological mapping benchmark

## 作者 / 提交者

- 姓名：李照阳（Zhaoyang Li）
- 单位：河南理工大学（Henan Polytechnic University）
- 邮箱：3286533389@qq.com
- ORCID（如有）：（可留空）

## 摘要 / 简介

本数据集包含 S2-LithoBench v1.0 的完整可复现流程与衍生数据表。该基准在澳大利亚两个干旱研究区（昆士兰州 Mount Isa Inlier 与南澳 Flinders Ranges）比较了用于岩性填图的特征工程、分类器与验证方案，全部使用公开数据：Sentinel-2 L2A 影像（Earth Search STAC，Element 84/AWS）、澳大利亚地质调查局 1:100 万地表地质图（CC BY 4.0）与 Copernicus DEM GLO-30（AWS）。

内容包括：8 个有序 Python 脚本（地质图下载、分层采样、多时相 Sentinel-2 特征提取、地形融合、模型对比、绘图、示范制图、统计分析与稳健性检验）；带全部特征的样本表（n = 2014 与 n = 1278）；指标表、混淆矩阵、袋外预测、置换重要性；10 次空间分块重复 + Wilcoxon/Holm 检验、缓冲区验证（0.5/1/2 km）、分块尺寸敏感性、地质单元 GroupKFold、类别均衡与逐单元限样、多数类与标签置换对照、XGBoost 结果、跨区迁移置信区间；运行日志与论文图件（PNG + PDF）。全部随机种子固定（SEED = 42）；在 16 核桌面、无 GPU 环境下完整流程约需 70 分钟。

主要发现：像素随机验证相对空间分块验证把宏 F1 虚高 0.16–0.41；地形属性带来唯一的大幅增量（+0.079/+0.094，仅地形特征达 0.199/0.279）；光谱指数与 3×3 纹理贡献不超过 0.03；按整条地质单元留出后宏 F1 降至 0.091/0.109，标签置换对照为 0.145/0.160；跨区迁移接近随机。

## 关键词

岩性填图；空间交叉验证；基准数据集；Sentinel-2；可复现性；模型可迁移性；干旱区；澳大利亚

## 学科分类

地球科学 → 地质学 / 遥感地质；计算机科学 → 机器学习

## 数据来源与采集方法

- Sentinel-2 L2A：Earth Search STAC API（Element 84 / AWS Open Data），2024 年旱季各 5 景，SCL 云掩膜，统一到 20 m 网格
- 参考地质：Geoscience Australia《Surface Geology of Australia》1:100 万 WMS（图层 AUS_GA_1M_GUPoly_Lithology，CC BY 4.0）
- 地形：Copernicus DEM GLO-30（AWS `copernicus-dem-30m`）

## 使用说明

1. 安装环境：`pip install -r requirements.txt`（Python 3.14）
2. 依次运行 `scripts/02_query_geology.py` → `03_make_samples.py` → `04_extract_features.py` → `05_train_models.py` → `06_figures.py` → `07_map_demo.py` → `08_analysis.py` → `09_robustness.py`
3. 脚本直接读取公开数据，无需任何密钥；训练模型文件（*.joblib，约 100 MB）未包含，可由 `05_train_models.py` 在约 13 分钟内重新生成

## 许可协议

- 代码：MIT License（见 `LICENSE`）
- 衍生数据表：CC BY 4.0（见 `LICENSE-DATA`）
- 上游影像与参考图仍遵循其原始许可

## 基金

无外部资助。

## 备注

发布后请把 DOI（形如 10.xxxxx/xxxxx）和数据集链接发给论文作者，用于回填论文的
"Data and code availability" 一节。
