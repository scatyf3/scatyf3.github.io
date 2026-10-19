---
title: "高效训练独立 Draft Model"
title_en: "Training Independent Draft Models Efficiently"
date: 2026-10-15
tags:
  - speculative decoding
  - training
---

<div class="lang-zh" markdown="1">

之前research的时候要在standalone draft上复现[\[2509.22134\] Bridging Draft Policy Misalignment: Group Tree Optimization for Speculative Decoding](https://arxiv.org/abs/2509.22134)，但是遇到俩问题

1. low gpu use被kill
2. gpu hours太多
   「All results were obtained on NVIDIA A100 80 GB GPUs under mixed precision with ZeRO-2. The Phase-II GTO fine-tuning requires approximately (i) 200 GPU-hours for 8B models, (ii) 400 GPU-hours for 13B models, and (iii) 900 GPU-hours for 70B models.」


GTO的设计是想解决eagle train(as chain) inference(as tree) mismatch的问题，在tree上train draft的能力，数学上的detail可以去paper里读，从infra的角度，highlevel的workflow是：

1. 在一个group里的k段xm个位置，各自rollout出一个tree
2. Target verify
3. 对比ref，计算advantage，然后计算一些神奇的rl loss
4. draft replay+backprop

![GTO workflow: rollout, verify, advantage, replay](/images/blog/gto-workflow.png)

做一下naive版codebase的time breakdown，配置如下：
A6000，Qwen3-8B + Qwen3-0.6B，bf16，SharedGPT generate response with target, avg sample_len = 297 token, 8 sampling, 每个data之间串行：

| ms/sample                                  | v0 eager + SDPA |
| ------------------------------------------ | --------------- |
| rollout（16 tree generate/96 draft forward） | 1269 (45.5%)    |
| verify（16 target forward）                  | 607 (21.7%)     |
| replay（16 draft forward）                   | 295 (10.6%)     |
| prefix + KL + loss                         | 118 (4.2%)      |
| backward                                   | 503 (18.0%)     |
| step time                                  | **2793**        |
| peak GPU memory                            | 24.9 GiB        |

### preliminary: tree

这里我们用的是动态树和静态树的混合做法：

1. tree generate，我们预定义每层的tree fanout的超参数，如 `[4, 3, 2, 2, 1]`，即第一层 1->4，第二层 4->12
2. pruning，用eagle2定义的draft accumulate probility prune掉概率小的tree node


### rollout runtime

这个是优化的起点，当时拿着在KV Cache eval的经验自告奋勇说要优化下draft rl training这边的rollout。非常显然，decoding时的kernel launch在v0是最大的bound。优化方法也同理，我们用static KV，用一个有valid length作为参数的kernel skip后面的计算，同时设置一个static的tree mask buffer，分不同的q len，分别CUDA Graph capture draft tree forward。

overhead的消除效果如下表所示：

| ms/sample                                  | eager + SDPA rollout | graph + triton static rollout |
| ------------------------------------------ | -------------------- | ----------------------------- |
| rollout（16 tree generate/96 draft forward） | 1269 (45.5%)         | 673 (30.1%)                   |
| verify（16 target forward）                  | 607 (21.7%)          | 601 (26.9%)                   |
| replay（16 draft forward）                   | 295 (10.6%)          | 313 (14.0%)                   |
| prefix + KL + loss                         | 118 (4.2%)           | 121 (5.4%)                    |
| backward                                   | 503 (18.0%)          | 529 (23.6%)                   |
| step time                                  | **2793**             | **2237**                      |
| peak GPU memory                            | 24.9 GiB             | 25.1 GiB                      |

### workflow

迁移完上面偏inference直觉的优化后，在agent老师的hint下，找到了一个重要的优化维度

现在rollout/verify/replay还是naive的串行，即对每个position，串行使用tree_generate生成一个tree，verify再replay，但这些tree之间没有依赖关系，可以通过batching和mask来做parallel

mask分为两部分

1. prefix kv，全部prompt+response的kv cache。我们mask一个rollout的suffix，让它只能看到这个位置之前的kv
2. forest kv，每个tree自己decode的结果，一个tree只看自己的prefix，mask掉其他tree和本tree 非prefix

![Forest attention mask](/images/blog/forest-mask.png)

这个forest mask的规则对rollout，verify和replay都起作用，contribute了本blog最大的speedup

| ms / sample    | L0（bs=1 ）  | L1（bs=4）   | L2（b=16）  |
| -------------- | ---------- | ---------- | --------- |
| rollout        | 665.1      | 190.3      | 77.4      |
| verify         | 601.7      | 398.8      | 345.2     |
| replay         | 312.5      | 84.2       | 69.2      |
| prefix+KL+loss | 119.5      | 92.5       | 87.6      |
| backward       | 565.2      | 246.9      | 297.6     |
| **step**       | **2264.2** | **1012.8** | **877.0** |

### Flex Attention

下一步是给target和draft也加一些高效kernel。

FlexAttention是torch新推出的功能，given一个自定义mask，torch会自己编译一个flash attention style的forward和backward，demo如下：
```python
from torch.nn.attention.flex_attention import flex_attention, create_block_mask

# build the tree mask
def tree_mask_mod(b, h, q_idx, kv_idx):
    in_prefix  = kv_idx < P
    see_prefix = kv_idx <= anchor_pos[q_idx]           # 只看到自己 anchor 之前的 prefix
    j  = (kv_idx - P).clamp(min=0)                     # kv 列对应的 forest 节点
    dj = depths[j]
    see_node = (dj <= depths[q_idx]) & (anc_at[q_idx, dj.clamp(max=gamma-1)] == j)
    return torch.where(in_prefix, see_prefix, see_node)

block_mask = create_block_mask(tree_mask_mod, B=None, H=None,
                               Q_LEN=M, KV_LEN=P + M, device="cuda")
flex = torch.compile(flex_attention, dynamic=False)
out = flex(q, k, v, block_mask=block_mask, kernel_options=KOPT)   # 支持 backward
```

forest replay without and with flex attention

| ms/sample              | wo flex attention（replay：SDPA + 稠密 mask） | w flex attention（replay：Flex + BlockMask） | 加速        |
| ---------------------- | ---------------------------------------- | ----------------------------------------- | --------- |
| rollout (Triton+Graph) | 76.3 (8.8%)                              | 75.9 (10.8%)                              | –         |
| verify (Triton+eager)  | 342.0 (39.4%)                            | 341.5 (48.5%)                             | –         |
| replay                 | 68.1 (7.8%)                              | 54.1 (7.7%)                               | 1.26×     |
| prefix + KL + loss     | 85.7 (9.9%)                              | 85.0 (12.1%)                              | –         |
| backward               | 296.3 (34.1%)                            | 147.6 (21.0%)                             | **2.01×** |
| **step time**          | **868.5**                                | **704.1**                                 | **1.23×** |

这里backward收益最大，因为attention backward的计算量本来就大，放大了flex attention 跳过mask部分的优势。

### summary

以上是我们优化的效果图：

![Step time overview](/images/blog/gto-step-time-overview.png)

还有在rtx 5070ti的demo结果：

todo

### limitations

这个project最开始的动机只是优化下tree rollout，后续又出于想学rl infra的目的快速迭代了一版。我们的research后续转向了off policy的loss formulation，我觉得也make sense，但从工程角度，这个codebase没人用过，所以只有数值在误差范围下的正确性保证，而没有e2e的正确性保证，未来的followup是

1. 运行这个repo，试着train一个draft。本来的motivation就是我assume scale up之后，grpo的效果好点
2. pr for specforge，现在有些rl train draft的方法，包括gto和opd for sd，可以提一波pr

如果读者觉得我们的paper（或者这个codebase）有帮助，please cite: 

```bibtex
@misc{xia2026acceptanceawaredraftmodeltraining,
      title={Acceptance-Aware Draft Model Training for Speculative Decoding}, 
      author={Tianhua Xia and Mugilan Ganesan and Yifei Feng and Haiyu Wang and Maximilian Egger and Sai Qian Zhang},
      year={2026},
      eprint={2609.24150},
      archivePrefix={arXiv},
      primaryClass={cs.LG},
      url={https://arxiv.org/abs/2609.24150}, 
}
```

</div>

<div class="lang-en" markdown="1">

During earlier research I tried to reproduce [\[2509.22134\] Bridging Draft Policy Misalignment: Group Tree Optimization for Speculative Decoding](https://arxiv.org/abs/2509.22134) on a standalone draft model, and ran into two problems:

1. The job got killed for low GPU utilization.
2. It needs far too many GPU hours.
   "All results were obtained on NVIDIA A100 80 GB GPUs under mixed precision with ZeRO-2. The Phase-II GTO fine-tuning requires approximately (i) 200 GPU-hours for 8B models, (ii) 400 GPU-hours for 13B models, and (iii) 900 GPU-hours for 70B models."

GTO is designed to fix the mismatch in EAGLE, which is trained as a chain but used as a tree at inference time, by training the draft directly on trees. The math is in the paper. From an infra point of view, the high-level workflow is:

1. For k segments × m positions in a group, roll out one tree at each position.
2. The target verifies the trees.
3. Compare against the reference, compute the advantage, then compute some fancy RL loss.
4. Draft replay + backprop.

![GTO workflow: rollout, verify, advantage, replay](/images/blog/gto-workflow.png)

First, a time breakdown of the naive codebase. Setup:
A6000, Qwen3-8B + Qwen3-0.6B, bf16, ShareGPT with responses generated by the target, avg sample_len = 297 tokens, 8 samples, data processed serially:

| ms/sample                                  | v0 eager + SDPA |
| ------------------------------------------ | --------------- |
| rollout (16 tree generate / 96 draft forward) | 1269 (45.5%) |
| verify (16 target forward)                 | 607 (21.7%)     |
| replay (16 draft forward)                  | 295 (10.6%)     |
| prefix + KL + loss                         | 118 (4.2%)      |
| backward                                   | 503 (18.0%)     |
| step time                                  | **2793**        |
| peak GPU memory                            | 24.9 GiB        |

### Preliminary: the tree

We use a hybrid of dynamic and static trees:

1. Tree generation: a per-layer fanout is set as a hyperparameter, e.g. `[4, 3, 2, 2, 1]`, so layer 1 expands 1→4, layer 2 expands 4→12, and so on.
2. Pruning: low-probability tree nodes are pruned using the accumulated draft probability defined in EAGLE-2.

### Rollout runtime

This was the starting point. Coming off the KV cache eval work, I volunteered to optimize the rollout side of draft RL training. Unsurprisingly, kernel launch overhead during decoding was the main bottleneck in v0. The fix is the same as for inference. We use a static KV cache and a kernel that takes the valid length as an argument and skips everything past it. We also allocate a static tree-mask buffer and capture the draft tree forward with CUDA Graphs, one graph per query length.

The effect of removing this overhead:

| ms/sample                                  | eager + SDPA rollout | graph + triton static rollout |
| ------------------------------------------ | -------------------- | ----------------------------- |
| rollout (16 tree generate / 96 draft forward) | 1269 (45.5%)      | 673 (30.1%)                   |
| verify (16 target forward)                 | 607 (21.7%)          | 601 (26.9%)                   |
| replay (16 draft forward)                  | 295 (10.6%)          | 313 (14.0%)                   |
| prefix + KL + loss                         | 118 (4.2%)           | 121 (5.4%)                    |
| backward                                   | 503 (18.0%)          | 529 (23.6%)                   |
| step time                                  | **2793**             | **2237**                      |
| peak GPU memory                            | 24.9 GiB             | 25.1 GiB                      |

### Workflow

After porting these inference-style optimizations, a hint from the agent pointed me to a more important dimension to optimize.

Rollout, verify and replay were still naively serial: for each position, generate one tree with tree_generate, verify it, then replay it. But the trees don't depend on each other, so they can run in parallel with batching and masking.

The mask has two parts:

1. Prefix KV: the KV cache of the full prompt + response. We mask the suffix for each rollout so it only sees the KV before its own position.
2. Forest KV: what each tree decodes itself. A tree only sees its own ancestors; other trees and the non-ancestor nodes of its own tree are masked out.

![Forest attention mask](/images/blog/forest-mask.png)

The same forest mask applies to rollout, verify and replay, and it gives the biggest speedup in this post.

| ms / sample    | L0 (bs=1)  | L1 (bs=4)  | L2 (bs=16) |
| -------------- | ---------- | ---------- | ---------- |
| rollout        | 665.1      | 190.3      | 77.4       |
| verify         | 601.7      | 398.8      | 345.2      |
| replay         | 312.5      | 84.2       | 69.2       |
| prefix+KL+loss | 119.5      | 92.5       | 87.6       |
| backward       | 565.2      | 246.9      | 297.6      |
| **step**       | **2264.2** | **1012.8** | **877.0**  |

### Flex Attention

The next step is to give the target and draft more efficient kernels.

FlexAttention is a recent PyTorch feature: given a custom mask, PyTorch compiles a FlashAttention-style forward and backward for it. A demo:

```python
from torch.nn.attention.flex_attention import flex_attention, create_block_mask

# build the tree mask
def tree_mask_mod(b, h, q_idx, kv_idx):
    in_prefix  = kv_idx < P
    see_prefix = kv_idx <= anchor_pos[q_idx]           # only see the prefix before its own anchor
    j  = (kv_idx - P).clamp(min=0)                     # forest node for this kv column
    dj = depths[j]
    see_node = (dj <= depths[q_idx]) & (anc_at[q_idx, dj.clamp(max=gamma-1)] == j)
    return torch.where(in_prefix, see_prefix, see_node)

block_mask = create_block_mask(tree_mask_mod, B=None, H=None,
                               Q_LEN=M, KV_LEN=P + M, device="cuda")
flex = torch.compile(flex_attention, dynamic=False)
out = flex(q, k, v, block_mask=block_mask, kernel_options=KOPT)   # supports backward
```

Forest replay without and with FlexAttention:

| ms/sample              | w/o FlexAttention (replay: SDPA + dense mask) | w/ FlexAttention (replay: Flex + BlockMask) | speedup   |
| ---------------------- | --------------------------------------------- | ------------------------------------------- | --------- |
| rollout (Triton+Graph) | 76.3 (8.8%)                                   | 75.9 (10.8%)                                | –         |
| verify (Triton+eager)  | 342.0 (39.4%)                                 | 341.5 (48.5%)                               | –         |
| replay                 | 68.1 (7.8%)                                   | 54.1 (7.7%)                                 | 1.26×     |
| prefix + KL + loss     | 85.7 (9.9%)                                   | 85.0 (12.1%)                                | –         |
| backward               | 296.3 (34.1%)                                 | 147.6 (21.0%)                               | **2.01×** |
| **step time**          | **868.5**                                     | **704.1**                                   | **1.23×** |

Backward gains the most. Attention backward is compute-heavy to begin with, which magnifies FlexAttention's advantage of skipping masked blocks.

### Summary

The overall effect of these optimizations:

![Step time overview](/images/blog/gto-step-time-overview.png)

Demo results on an RTX 5070 Ti:

todo

### Limitations

This project started out only as an attempt to speed up tree rollout, and I then quickly iterated on it because I wanted to learn RL infra. Our research later moved to an off-policy loss formulation, which I think also makes sense. From an engineering standpoint, though, nobody has used this codebase yet: correctness is only checked numerically, within error tolerance, not end to end. Follow-ups:

1. Run this repo and actually train a draft. The original motivation was my assumption that GRPO works better once scaled up.
2. PRs to SpecForge. There are now several RL-based draft training methods, including GTO and OPD for SD, worth upstreaming.

If you find our paper (or this codebase) useful, please cite:

```bibtex
@misc{xia2026acceptanceawaredraftmodeltraining,
      title={Acceptance-Aware Draft Model Training for Speculative Decoding}, 
      author={Tianhua Xia and Mugilan Ganesan and Yifei Feng and Haiyu Wang and Maximilian Egger and Sai Qian Zhang},
      year={2026},
      eprint={2609.24150},
      archivePrefix={arXiv},
      primaryClass={cs.LG},
      url={https://arxiv.org/abs/2609.24150}, 
}
```

</div>
