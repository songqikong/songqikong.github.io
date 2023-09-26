---
layout: post
title: A Brief Study note of "SCoDA"
subtitle: SCoDA: Domain Adaptive Shape Completion for Real Scans, CVPR 2023
tags: [paper reading, deep learning, point cloud completion]
toc: true
---



# Ⅰ. Abstract

> 3D shape completion from point clouds is a challenging task, especially from scans of real-world objects. Considering the paucity of 3D shape ground truths for real scans, existing works mainly focus on benchmarking this task on synthetic data, e.g. 3D computer-aided design models. However, the domain gap between synthetic and real data limits the generalizability of these methods. Thus, we propose a new task, SCoDA, for the domain adaptation of real scan shape completion from synthetic data. A new dataset, ScanSalon, is contributed with a bunch of elaborate 3D models created by skillful artists according to scans. To address this new task, we propose a novel cross-domain feature fusion method for knowledge transfer and a novel volume-consistent self-training framework for robust learning from real data. Extensive experiments prove our method is effective to bring an improvement of 6%∼7% mIoU.

文章认为从真实世界扫描到的的三维形状补全是一个很有挑战性的任务，目前的工作仅仅集中在对合成数据进行基准测试（如从CAD模型中模拟扫描过程获取的点云），但是真实扫描数据和合成数据的差异性限制了这些工作的泛化性能；因此文章提出了一个新的任务：**SCoDa**（Domain Adaptive ShapeCompletion），即**对从合成数据出发对真实扫描形状进行补全的域自适应**。文章提出了一个新的数据集：**ScanSalon**（realScans with Shape manual annotations），是人工从扫描的数据中创造的3D模型；又提出了一种新的跨域特征融合方法用于知识转移，以及一种新的体积一致的自训练框架，用于从真实数据中进行鲁棒学习。



# Ⅱ. Contributions

- 提出了一种新的任务 **SCoDA**，即真实扫描的域自适应形状补全；贡献了一个包含 800 个精细 3D 模型的数据集**ScanSalon**，并与真实扫描的 6 个类别一一对应。

- 设计了一种新的跨域特征融合模块（cross-domain feature fusion module），将合成域和实域学习到的全局形状和局部模式的知识结合起来。这种特征融合方式可能给二维域适应领域（2D domain adaption field）的工作带来灵感。

  > For an effective transfer, we observe that although the local patterns of real scans (e.g., noise, sparsity, and incompleteness) are distinct from the simulated synthetic ones, the global topology or structure in a same category is usually similar between the synthetic and real data, for example, a chair from either the synthetic or real domain usually consists of a seat, a backrest, legs, etc. (see Fig. 1). In other words, the global topology is more likely to be domain invariant, while the local patterns are more likely to be domain specific.

- 提出了一种体积一致的自训练框架（volume-consistent self-training framework），以提高形状补全对真实扫描复杂不完整性的鲁棒性。
- 为基于**ScanSalon**的**SCoDA**任务构建了多种评估方法的基准;大量的实验也证明了所提方法的优越性。

# Ⅲ. Method

IF-Nets 是一种有前途的重建方法，本文就是基于IF-Nets进行的改进。我们的方法首先通过提出一种**新的跨域特征融合模块**来改进表示学习，该模块旨在将标签丰富的合成域中学习到的全局级对象形状的知识转移到真实域中。其次，为了在真实数据中耗尽特定领域的信息，针对特定的形状完成任务提出了一种新的**体积一致的自训练方法**。

## 3.1. Implicit Feature Networks

我们首先介绍IF-Nets[Implicit Functions in Feature Space for 3D Shape Reconstruction and Completion]作为我们的重建框架。IF-Net 由用于多尺度特征提取的 3D 卷积神经网络编码器 $g(\cdot)$ 和用于隐式形状解码的多层感知器组成。

给出一个点云样本 $P,$ 首先将他转换为一个体素的表示 $\textbf{X }\in\:\mathbb{R}^{\boldsymbol{N}\times\boldsymbol{N}\times\boldsymbol{N}}$ ，其中
$\begin{aligned}N&\in\:\mathbb{N}\end{aligned}$ 是输入空间的分辨率。$\text{X}$ 被送入一个L层的感知机 $g(\cdot)$ 来生成多尺度的特征 $\{\mathbf{F}_{1},....,\mathbf{F}_{L}\}$ ，然后他们被上采样到相同的空间维度并沿着channel进行拼接来生成最终特征 $\mathbf{F}=$concat( {upsa $\mathbf{concat}(\{\mathbf{upsample}(\mathbf{F}_{1}),[USD3P],$ $\mathbf{mple}(\mathbf{F}_{1}),[USD3P],\mathbf{upsample}(\mathbf{F}_{L})\})$. Formally:
$$
g(\mathbf{X})=\mathbf{F},\quad\mathbf{X}\in\mathbb{R}^{N\times N\times N},\quad\mathbf{F}\in\mathbb{R}^{d\times N\times N\times N}
$$

where $\text{a}$ is the feature channel number that equals the sum-mation of the channel numbers of $\mathbf{F}_{\boldsymbol{i}}$ .Note that $\text{H}$ has a3D structure aligned with the input $\text{X}$ . Given a point query
$\mathbf{p}\in\mathbb{R}^{3}$, continuous features $\mathbf{F}(\mathbf{p})$ at this point can be ex-tracted from $\text{F}$ using trilinear interpolation.

Then, the encoding at the point $\text{P}$ is fed into a point-wise

decoder $f(\cdot)$ to give a binary prediction indicating if the point lies inside or outside the shape:

$$
f(\mathbf{F},\mathbf{p})=f(\mathbf{F}(\mathbf{p}))\mapsto\{0,1\}.
$$

Given the occupancy value $o(\mathbf{p})\in\{0,1\}$ at each position p pre-computed according to the ground truth shape mesh.a binary cross-entropy loss is used to train $\text{9}$ and $\text{f}$

$$
\min L_{IF}=\text{BCE}\big(f(\mathbf{F},\mathbf{p}),\:o(\mathbf{p})\big).
$$

Note that the prediction values can also be continuous signed distance values as in [58].
