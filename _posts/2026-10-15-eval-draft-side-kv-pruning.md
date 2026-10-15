---
title: "How to Evaluate Draft-Side KV Cache Pruning"
date: 2026-10-15
excerpt: "🚧 Work in progress, coming soon."
tags:
  - speculative decoding
  - KV cache
---

> 🚧 **Work in progress.** This post is being written and will be updated soon.

<!-- TODO: 一句话 TL;DR：评估 draft-side KV pruning 要看什么、最容易踩的坑是什么 -->

## Why draft-side pruning is different

<!-- TODO:
- 普通 KV pruning 评估的是 target 的输出质量（accuracy / perplexity）
- draft-side pruning 不改变 target 输出（verify 保证无损），所以质量指标失效，真正要看的是速度
- 核心 trade-off：draft 每步更便宜 vs. acceptance length 下降
-->

## Metrics

<!-- TODO:
- Acceptance length（mean accepted tokens per step）
- Draft step latency / draft 占总时间的比例
- End-to-end speedup（tokens/s，相对 no-spec 和 full-KV draft）
- 不同 KV budget 下的曲线，而不是单点
-->

## Workloads

<!-- TODO:
- 为什么要用长推理 trace：短 prompt 下 draft KV 本来就小，剪了也看不出差别
- 数据集选择：
- 生成长度分布：
-->

## Baselines

<!-- TODO:
- Full draft KV（上界参考）
- StreamingLLM / sliding window
- H2O / SnapKV 等 target-side 方法直接搬到 draft 上
-->

## Pitfalls

<!-- TODO:
- 只报 acceptance length 不报 wall-clock
- batch size 1 vs. batched：加速在 batching 下会缩水
- temperature / sampling 设置对 acceptance 的影响
- pruning 本身的开销有没有算进去
-->

## Our setup and results

<!-- TODO: 实验配置、主要结果图 -->

## Takeaways

<!-- TODO -->
