---
title: "分布式 Speculative Decoding"
title_en: "Distributed Speculative Decoding"
date: 2026-10-17
excerpt: "🚧 Work in progress, coming soon."
tags:
  - speculative decoding
  - distributed inference
---

<div class="lang-zh" markdown="1">

> 🚧 **施工中。** 这篇还在写，很快更新。

<!-- TODO: 一句话 TL;DR：draft 和 target 不在同一张卡 / 同一台机器上时，SD 的瓶颈变成了什么 -->

### 动机

<!-- TODO:
- 为什么要把 draft 和 target 拆开部署：显存、异构硬件、draft 跨 target 复用……
- 和单卡 SD 的区别
-->

### 设置

<!-- TODO:
- draft / target 各自放在哪（同机多卡、跨机、TP / PP 下的 target）
- 模型组合、硬件、网络带宽
-->

### 瓶颈在哪

<!-- TODO:
- 每一轮 draft → verify → accept 的通信量与延迟
- 通信 vs. 计算的占比，什么时候通信 dominate
-->

### 优化

<!-- TODO:
- draft 和 verify 的 overlap / pipelining
- 减少 round trip：多步 draft、tree、异步 verify
-->

### 实验

<!-- TODO: 吞吐 / 延迟 / acceptance length，对比单卡 SD 和 target AR -->

### 参考

<!-- TODO: 相关论文与代码链接 -->

</div>

<div class="lang-en" markdown="1">

> 🚧 **Work in progress.** This post is being written and will be updated soon.

<!-- TODO: one-line TL;DR -->

### Motivation

<!-- TODO -->

### Setup

<!-- TODO -->

### Where the bottleneck is

<!-- TODO -->

### Optimizations

<!-- TODO -->

### Experiments

<!-- TODO -->

### References

<!-- TODO -->

</div>
