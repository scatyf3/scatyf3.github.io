---
title: "如何评估 Draft-Side KV Cache Pruning"
date: 2026-10-15
tags:
  - speculative decoding
  - KV cache
---

### 动机与约束

我们想通过KV Cache Pruning优化standalone draft在Long Decoding时的效率。Why Standalone Draft？我的理解是这个做法在生产部署中比较常见，虽然不如dflash和eagle那么流行，原因有二：

1. 让 draft 和 target 重叠执行的系统（例如 speculative speculative decoding）很难使用 target 的 hidden states；
2. target 的权重可能有访问限制，依赖 target 的 draft 训练在流程上很难推进。

### 理论分析

在 standalone draft 的 long decoding 里，我们发现 draft side 的 KV cache memory traffic 逐渐成为瓶颈。考虑一个 single request 的场景，每一轮 speculation 中：

1. draft：
	1. KV：$\gamma \times \text{ctx\_len} \times L_{d}\times H_{d} \times D_{d} \times 2 \times \text{sizeof(BF16)}$ 
	2. weight：$\gamma \times W_{draft}$
2. target：
	1. KV：$1 \times  \text{ctx\_len} \times L_{t}\times H_{t} \times D_{t} \times 2 \times \text{sizeof(BF16)}$
	2. weight：$W_{target}$
3. 定性来看：每个 token 的 KV 大小是 $L \times H_{kv} \times D$，target 的 $L_t, H_{t}, D_t$ 一般比 draft 大，但差距远小于两者参数量的差距；而 draft 每轮要 decode $\gamma$ 步，因此两者斜率之比为 $\dfrac{\gamma \, L_d H_d D_d}{L_t H_t D_t}$，通常大于 1。


以 Qwen3-0.6B + 8B 为例，$H_{kv}=8$、$D=128$ 相同，只有层数不同（28 vs 36），$\gamma=7$ 时斜率之比为 $7 \times 28 / 36 \approx 5.4$。这样我们可以用 model architecture 所记录的常数，把 target 和 draft 的 memory traffic 画成关于 context length 的一次函数。可以发现在 10K token 之后，draft 的 memory traffic 开始大于 target，所以这里有很大的优化空间：

![Memory traffic per speculation round for Qwen3-0.6B draft vs. Qwen3-8B target](/images/blog/draft-target-mem-traffic.svg)
*图 1：每一轮 speculation 的 HBM traffic（single request，BF16）。context 超过约 10K token 后，draft 的 γ 次 forward 读取的字节数超过 target 的一次 verify。*

### Eval Pipeline Building

事实上，在这篇 work 里遇到的最大困难是如何构建 eval pipeline，无论是从 speculative decoding 的角度，还是从 KV cache pruning 的角度。

#### Speculative decoding

开始做 standalone SD 的时候，最经典的操作就是用 SpecInfer 训练的 llama-68m + Llama-2-7B 做 SD，这组模型在 Hugging Face / PyTorch 的实现里效果很不错。然而同样的实现，切换到 Qwen3-0.6B + 8B，效果变得非常烂：SD 和 target AR 的速度近似。于是开始了一通排查：

1. **算法实现错了？** acceptance length 在 3～4，理论上会有加速。
2. **性能有问题？** 用 torch profiler 排查，发现 draft model 的 CPU timeline 打满，GPU timeline 空转，其实是 CPU side 的问题（kernel launch bound）。

![torch profiler timeline of a decode step, Qwen3-0.6B vs. 14B](/images/blog/draft-profiler-cpu-bound.png)
*图 2：一个 decode step 的 torch profiler trace。下方 GPU 一行的 kernel 都很短、彼此之间有空隙，上方 CPU 一侧一直在 launch op。*

#### KV cache pruning

在 KV cache pruning 部分也遇到了这个问题：完全按照一些 pruning 方法的代码实现后，发现 prune draft KV cache 前后没有什么特别显著的性能差距。

1. 还是之前说的 CPU launch bound。
2. KV cache pruning 的 paper 用相同的实现却会有加速，因为它们的 evaluation 设置是：
    1. 从超长 context 开始；
    2. large batching。

    所以 attention dominate 了 decoding process，cover 掉了 CPU launch bound。

#### CUDA Graph

了解这个 bound 之后，很显然的优化是用 CUDA Graph 来减少 kernel launch 开销。但当时并不熟悉 CUDA Graph 怎么合理使用：vLLM / SGLang 的 paper 里提及的是最核心的 serving 机制，如 paged / radix attention、continuous batching 等等，鲜有 paper 讨论好用、基础、但可能缺乏 novelty 的 CUDA Graph，所以实现的时候也遇到了一些波折。

CUDA Graph 的基本原理很简单：提前 record 好需要 launch 的 kernel 序列和 input，捕获成一个 graph，真正 inference 的时候 replay 这个 graph 即可。当然，这样的捕获要求 kernel 的 input memory buffer 不变，否则需要触发 recapture。

#### v1：预分配 KV + masked SDPA

最开始的 v1 evaluation pipeline 是仿照 TriForce 的 codebase 改的一版最 naive 的实现。简而言之：在 prefill 之后，把 KV 压缩到一个固定的 `prefill_budget`，然后再 decode 固定的 `max_new_token`。为了兼容 CUDA Graph，我们 preallocate `prefill_budget + max_new_token` 大小的 KV cache，使用一个统一 input shape 的 SDPA attention，mask 掉后面的 invalid 内容。伪代码如下：

```python
k_cache = torch.zeros(L, B, H_kv, S_MAX, D)              # 每层一份
v_cache = torch.zeros(L, B, H_kv, S_MAX, D)
valid   = torch.zeros(B, 1, 1, S_MAX, dtype=torch.bool)   # bool mask，True = 有效
pos     = torch.zeros(B, dtype=torch.long)                # 当前写入位置，用张量不用 int

o = F.scaled_dot_product_attention(
    q, k_cache[l], v_cache[l],                            # 永远是完整的 S_MAX 长度
    attn_mask=valid,                                      # 屏蔽还没写入的位置
    enable_gqa=True,
)
```

#### v1 的问题

这个方法在 eval long prefilling 时没有问题，比如做 NIAH 或文档问答，`max_new_token` 很少，mask 的浪费非常小。然而如果 eval long decoding，假设 prefill 1K、decode 32K，我们需要 preallocate 33K，然后 mask 掉尺寸非常大的 invalid KV，这显然不 make sense。

首先想到的解法是分成多个 graph，分别 record 1K～2K、2K～4K、4K～8K 的 kernel 来减少 mask 浪费。然而这个办法有点傻，而且并未真正解决浪费。和 agent 老师调研一番之后，决定改 kernel。

#### v2: varlen kernel

与其在 CPU side 构建 mask，不如直接改 kernel：kernel 直接读取 valid pos，skip 掉后面的计算。用一个最简单的 linear add 演示，比起最普通的 add kernel，我们额外传入 `len_ptr`，定义 `[0, len_ptr)` 才是有效数据，后面都是无效数据：

```python
@triton.jit
def add_kernel(a_ptr, b_ptr, out_ptr, len_ptr, BLOCK: tl.constexpr):
    pid = tl.program_id(0)
    start = pid * BLOCK
    valid_len = tl.load(len_ptr)          # 从显存读有效长度：graph replay 时读到的是最新值
    if start >= valid_len:                # 整个 block 都在无效区：直接退出，不读 a / b
        return
    offs = start + tl.arange(0, BLOCK)
    m = offs < valid_len                  # 跨越边界的那个 block：只处理有效部分
    a = tl.load(a_ptr + offs, mask=m)
    b = tl.load(b_ptr + offs, mask=m)
    tl.store(out_ptr + offs, a + b, mask=m)
```

更多 kernel 细节见 [IndepDraftEval/indep_draft/kernels/triton_varlen_attn.py](https://github.com/scatyf3/IndepDraftEval/blob/main/indep_draft/kernels/triton_varlen_attn.py)。


#### When to pruning

搭建好kernel和sd的基础设施(draft - verify - draft状态回退)，我们获得了我们想优化的baseline。接下来的问题就是在哪里做pruning。production时用的flash infer/flash attention没有中间的attention score。基于这个限制，做上古风格的，per step pruning的attention score based method是不行的，但也不能完全query无关，因为也要eval一些attention score的baseline。

基于这些限制，我们找到了R-KV，这是第一篇eval kv cache for long decoding的paper，提出了一些prune long decoding的方法，其流程翻译到我们的cuda是：
0. 接收一个short prompt
1. 正常deocde直到kv cache触及上限
2. 触发一次pruning，评分并压缩kv到下限，然后继续decoding


翻译到我们的cuda graph的runtime，这里的workflow是：
0. 接收一个short prompt
1. 正常使用flash attention style的kernel 高效decode，直到kv cache触及上限
2. 触及上限后，触发一次pruning，在pruning event里可以获得各种score，包括attention和其他fancy的algorithm，这里可以用eager的，非高效的实现，因为这个event触发频率很低
3. 按照score select好kv对，重新组织kv，把他们在物理地址上重新变得连续


我们的runtime比起上文的full kv sd baseline，需要增加：
1. 一个scoring接口，兼容各种评分方法
2. 一个 select + compact 步骤，按分数选出保留的 KV，并在原地压缩到 cache 前部，这里可以有个kernel，但profile下来latency占比非常少，所以保持torch原生。

综上，我们的eval pipeline如下：

![Continuous KV cache pruning overview](/images/blog/continuous-kv-pruning.svg)
*图 3：continuous KV cache pruning。KV cache 增长到上限 U 时触发一次 pruning：保留 sink 和 recent，在中间的 prunable 区域按 score 选出要保留的 KV，压缩到下限 L 后继续 decode。*


### Extend

截止到这里，我们只build了qwen3的pipeline，我们的方法该如何接入各类新模型及其新attention？如何支持tree speculative decoding？

#### SWA

t

Code：[lcrkv/kv_cache.py](https://github.com/scatyf3/lcrkv/blob/paper/lcrkv/kv_cache.py)（`RingScratchLayer`、`force_hybrid_ring_layers`）



#### Gated Delta Net

Code：[lcrkv/spec_decode/gdn_spec.py](https://github.com/scatyf3/lcrkv/blob/paper/lcrkv/spec_decode/gdn_spec.py)、[lcrkv/kernels/gdn_spec_kernel.py](https://github.com/scatyf3/lcrkv/blob/paper/lcrkv/kernels/gdn_spec_kernel.py)

#### Tree Speculative Decoding

Code：[indep_draft/tree_sd/](https://github.com/scatyf3/IndepDraftEval/tree/main/indep_draft/tree_sd)、[indep_draft/sd/tree.py](https://github.com/scatyf3/IndepDraftEval/blob/main/indep_draft/sd/tree.py)、[indep_draft/kernels/triton_varlen_tree.py](https://github.com/scatyf3/IndepDraftEval/blob/main/indep_draft/kernels/triton_varlen_tree.py)
