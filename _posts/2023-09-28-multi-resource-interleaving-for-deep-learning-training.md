---
layout: post
title: "深度学习训练的多源交错"学习笔记
subtitle: Multi-Resource Interleaving for Deep Learning Training
tags: [paper reading, deep learning, point cloud completion]
---

# Ⅰ. Abstract

> Training Deep Learning (DL) model requires multiple resource types, including CPUs, GPUs, storage IO, and network IO. Advancements in DL have produced a wide spectrum of models that have diverse usage patterns on different resource types. Existing DL schedulers focus on only GPU allocation, while missing the opportunity of packing jobs along multiple resource types. We present Muri, a multi-resource cluster scheduler for DL workloads. Muri exploits multi-resource interleaving of DL training jobs to achieve high resource utilization and reduce job completion time (JCT). DL jobs have a unique staged, iterative computation pattern. In contrast to multi-resource schedulers for big data workloads that pack jobs in the space dimension, Muri leverages this unique pattern to interleave jobs on the same set of resources in the time dimension. Muri adapts Blossom algorithm to find the perfect grouping plan for single-GPU jobs with two resource types, and generalizes the algorithm to handle multi-GPU jobs with more than two types. We build a prototype of Muri and integrate it with PyTorch. Experiments on a cluster with 64 GPUs demonstrate that Muri improves the average JCT by up to 3.6× and the makespan by up to 1.6× over existing DL schedulers.

训练深度学习 (DL) 模型需要多种资源类型，包括 CPU、GPU、存储 IO 和网络 IO等等。深度学习的进步催生了了大量的模型，这些模型在不同的资源类型上具有不同的使用模式。现有的DL调度器只关注GPU的分配，却错过了将作业整合到多个资源类型上的机会。

因此我们提出了 **Muri**，一种用于 DL 工作负载的多资源集群调度器。**Muri**利用DL训练作业的多资源交错来实现高资源利用率，并减少作业完成时间(JCT)。DL作业具有独特的分阶段迭代计算模式。与在空间维度上打包作业的大数据工作负载的多资源调度器不同，**Muri**利用这种独特的模式在时间维度上对同一组资源上的作业进行交错。**Muri**采用Blossom算法为**单单GPU作业**作业寻找两种资源类型的的完美分组方案，并将该算法推广到处理两种以上资源类型的**多GPU**作业。我们构建了一个**Muri**的原型，并将其与PyTorch集成。在具有64个GPU的集群上进行的实验表明，与现有的DL调度器相比，Muri将平均JCT提高了3.6倍，最大完成时间提高了1.6倍。

**CCS CONCEPTS**

- Computer systems organization → Cloud computing
- Computing methodologies → Machine learning.

# Ⅱ. Introduction

深度学习(DL)越来越多地集成到以数据为中心的互联网应用和服务中[8,20]。训练深度学习模型已经成为数据中心的一项重要工作