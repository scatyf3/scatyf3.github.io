---
title: "Training Independent Draft Models Efficiently"
date: 2026-10-15
excerpt: "🚧 Work in progress, coming soon."
tags:
  - speculative decoding
  - training
---

> 🚧 **Work in progress.** This post is being written and will be updated soon.

<!-- TODO: 一句话 TL;DR：训练独立 draft model 的主要开销在哪、怎么省 -->

## Independent drafts vs. draft heads

<!-- TODO:
- 独立 draft model（小 LM） vs. EAGLE / Medusa 这类挂在 target 上的 head
- 什么时候选独立 draft：跨 target 复用、部署灵活、……
-->

## What the objective should be

<!-- TODO:
- 目标是和 target 对齐（acceptance），不是 draft 自己的 accuracy
- 蒸馏：用 target 生成的数据 / target logits
- 链接到 acceptance-aware training 的论文
-->

## Where the cost goes

<!-- TODO:
- target 前向：online 跑 target vs. offline 预先缓存 logits / hidden states
- 大词表 logits 的显存（vocab × seq_len）
- 长序列训练数据
-->

## Making it efficient

<!-- TODO:
- Offline 数据生成与缓存，存储格式和体积
- Fused / chunked cross-entropy & KL kernel，避免物化完整 logits
- Sequence packing
- 并行策略和 runtime 的选择
-->

## Evaluating the draft

<!-- TODO:
- 训练中用什么 proxy 指标跟踪 acceptance
- 最终评估：acceptance length + end-to-end speedup
-->

## Takeaways

<!-- TODO -->
