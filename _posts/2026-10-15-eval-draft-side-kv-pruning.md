---
title: "如何评估 Draft-Side KV Cache Pruning"
title_en: "How to Evaluate Draft-Side KV Cache Pruning"
date: 2026-10-15
tags:
  - speculative decoding
  - KV cache
---

<div class="lang-zh" markdown="1">
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

</div>

<div class="lang-en" markdown="1">

### Motivation and Constraints

We want to use KV cache pruning to make a standalone draft model more efficient during long decoding. Why a standalone draft? My understanding is that this setup is fairly common in production deployments, even if it is less popular than DFlash and EAGLE, for two reasons:

1. Systems that overlap draft and target execution (e.g., speculative speculative decoding) can hardly use the target's hidden states;
2. The target's weights may have access restrictions, so draft training that depends on the target is hard to push through in practice.

### Theoretical Analysis

In long decoding with a standalone draft, we find that the draft-side KV cache memory traffic gradually becomes the bottleneck. Consider a single-request scenario. In each speculation round:

1. Draft:
	1. KV: $\gamma \times \text{ctx\_len} \times L_{d}\times H_{d} \times D_{d} \times 2 \times \text{sizeof(BF16)}$
	2. Weights: $\gamma \times W_{draft}$
2. Target:
	1. KV: $1 \times  \text{ctx\_len} \times L_{t}\times H_{t} \times D_{t} \times 2 \times \text{sizeof(BF16)}$
	2. Weights: $W_{target}$
3. Qualitatively: the per-token KV size is $L \times H_{kv} \times D$. The target's $L_t, H_{t}, D_t$ are usually larger than the draft's, but the gap is far smaller than the gap in parameter count; meanwhile the draft decodes $\gamma$ steps per round, so the ratio of the two slopes is $\dfrac{\gamma \, L_d H_d D_d}{L_t H_t D_t}$, which is usually greater than 1.

Take Qwen3-0.6B + 8B as an example: $H_{kv}=8$ and $D=128$ are the same, only the layer counts differ (28 vs 36). With $\gamma=7$, the slope ratio is $7 \times 28 / 36 \approx 5.4$. Using the constants from the model architecture, we can plot the memory traffic of target and draft as linear functions of context length. Beyond 10K tokens, the draft's memory traffic exceeds the target's, so there is a lot of room for optimization:

![Memory traffic per speculation round for Qwen3-0.6B draft vs. Qwen3-8B target](/images/blog/draft-target-mem-traffic.svg)
*Figure 1: HBM traffic per speculation round (single request, BF16). Beyond roughly 10K tokens of context, the bytes read by the draft's γ forward passes exceed those read by one target verify.*

### Eval Pipeline Building

In fact, the biggest difficulty in this work was building the eval pipeline, both from the speculative decoding side and from the KV cache pruning side.

#### Speculative decoding

When starting with standalone SD, the classic move is to run SD with SpecInfer's llama-68m + Llama-2-7B, which works very well in Hugging Face / PyTorch implementations. However, with the same implementation, switching to Qwen3-0.6B + 8B gave terrible results: SD ran at roughly the same speed as target AR. So the debugging began:

1. **Is the algorithm implemented wrong?** Acceptance length is 3–4, so in theory there should be a speedup.
2. **Is it a performance problem?** Profiling with the torch profiler showed the draft model's CPU timeline fully saturated while the GPU timeline sat idle. It was actually a CPU-side problem (kernel launch bound).

![torch profiler timeline of a decode step, Qwen3-0.6B vs. 14B](/images/blog/draft-profiler-cpu-bound.png)
*Figure 2: torch profiler trace of one decode step. The kernels in the GPU row at the bottom are all short with gaps between them, while the CPU side at the top is constantly launching ops.*

#### KV cache pruning

We hit the same problem in the KV cache pruning part: after implementing some pruning methods exactly following their code, there was no notable performance difference before and after pruning the draft KV cache.

1. Again, the CPU launch bound mentioned above.
2. KV cache pruning papers see speedups with the same implementation because their evaluation setup is:
    1. Start from a very long context;
    2. Large batching.

    So attention dominates the decoding process and hides the CPU launch bound.

#### CUDA Graph

Once we understood this bound, the obvious optimization was to use CUDA Graph to cut kernel launch overhead. But at the time I wasn't familiar with how to use CUDA Graph properly: the vLLM / SGLang papers describe the core serving mechanisms such as paged / radix attention and continuous batching, and few papers discuss CUDA Graph, which is useful and fundamental but perhaps lacks novelty. So the implementation had some twists and turns.

The basic idea of CUDA Graph is simple: record the sequence of kernels to launch and their inputs ahead of time, capture them as a graph, and replay that graph during actual inference. Of course, such capture requires the kernels' input memory buffers to stay fixed; otherwise a recapture is needed.

#### v1: Preallocated KV + masked SDPA

The initial v1 evaluation pipeline was the most naive implementation, adapted from the TriForce codebase. In short: after prefill, compress the KV to a fixed `prefill_budget`, then decode a fixed `max_new_token`. To be compatible with CUDA Graph, we preallocate a KV cache of size `prefill_budget + max_new_token` and use SDPA attention with a uniform input shape, masking out the invalid tail. Pseudocode:

```python
k_cache = torch.zeros(L, B, H_kv, S_MAX, D)              # one per layer
v_cache = torch.zeros(L, B, H_kv, S_MAX, D)
valid   = torch.zeros(B, 1, 1, S_MAX, dtype=torch.bool)   # bool mask, True = valid
pos     = torch.zeros(B, dtype=torch.long)                # current write position, a tensor not an int

o = F.scaled_dot_product_attention(
    q, k_cache[l], v_cache[l],                            # always the full S_MAX length
    attn_mask=valid,                                      # mask out positions not yet written
    enable_gqa=True,
)
```

#### Problems with v1

This approach is fine for evaluating long prefilling, e.g. NIAH or document QA, where `max_new_token` is small and the masking waste is tiny. But for long decoding, say 1K prefill and 32K decode, we need to preallocate 33K and then mask out a very large invalid KV region, which clearly doesn't make sense.

The first fix that came to mind was splitting into multiple graphs, recording kernels for 1K–2K, 2K–4K and 4K–8K separately to reduce the masking waste. But that is a bit clumsy and doesn't really eliminate the waste. After some research with my agent, I decided to change the kernel.

#### v2: varlen kernel

Rather than building a mask on the CPU side, change the kernel directly: the kernel reads the valid position itself and skips the computation beyond it. Using the simplest linear add as a demo: compared with a plain add kernel, we additionally pass `len_ptr`, defining `[0, len_ptr)` as valid data and everything after it as invalid:

```python
@triton.jit
def add_kernel(a_ptr, b_ptr, out_ptr, len_ptr, BLOCK: tl.constexpr):
    pid = tl.program_id(0)
    start = pid * BLOCK
    valid_len = tl.load(len_ptr)          # read the valid length from device memory: graph replay sees the latest value
    if start >= valid_len:                # the whole block is in the invalid region: exit without reading a / b
        return
    offs = start + tl.arange(0, BLOCK)
    m = offs < valid_len                  # the block that straddles the boundary: only process the valid part
    a = tl.load(a_ptr + offs, mask=m)
    b = tl.load(b_ptr + offs, mask=m)
    tl.store(out_ptr + offs, a + b, mask=m)
```

For more kernel details, see [IndepDraftEval/indep_draft/kernels/triton_varlen_attn.py](https://github.com/scatyf3/IndepDraftEval/blob/main/indep_draft/kernels/triton_varlen_attn.py).

#### When to prune

With the kernel and SD infrastructure in place (draft – verify – draft state rollback), we had the baseline we wanted to optimize. The next question is where to prune. The FlashInfer / FlashAttention kernels used in production don't expose intermediate attention scores. Given this constraint, old-style per-step pruning based on attention scores is not an option, but the method can't be completely query-agnostic either, because we also need to evaluate some attention-score baselines.

Under these constraints we found R-KV, the first paper to evaluate KV cache methods for long decoding, which proposes several ways to prune during long decoding. Its procedure is:

0. Receive a short prompt
1. Decode normally until the KV cache hits the upper limit
2. Trigger one pruning pass that scores the KV and compresses it to the lower limit, then continue decoding

Translated into our CUDA Graph runtime, the workflow is:

0. Receive a short prompt
1. Decode efficiently with a FlashAttention-style kernel until the KV cache hits the upper limit
2. On hitting the limit, trigger a pruning event. Inside the event we can obtain all kinds of scores, including attention and other fancy algorithms; an eager, inefficient implementation is fine here because the event fires rarely
3. Select KV pairs by score and reorganize the KV so the kept entries are physically contiguous again

Compared with the full-KV SD baseline above, our runtime adds:

1. A scoring interface that accommodates various scoring methods
2. A select + compact step that picks the KV to keep by score and compacts it in place to the front of the cache. This could be a kernel, but profiling shows it takes a tiny share of latency, so it stays in native torch.

Putting it all together, our eval pipeline looks like this:

![Continuous KV cache pruning overview](/images/blog/continuous-kv-pruning.svg)
*Figure 3: continuous KV cache pruning. When the KV cache grows to the upper limit U, one pruning pass fires: sink and recent tokens are kept, KV in the prunable middle region is selected by score, and the cache is compacted to the lower limit L before decoding continues.*

### Extend

So far we have only built the Qwen3 pipeline. How should our method plug into newer models and their new attention variants? How do we support tree speculative decoding?

#### SWA

t

Code: [lcrkv/kv_cache.py](https://github.com/scatyf3/lcrkv/blob/paper/lcrkv/kv_cache.py) (`RingScratchLayer`, `force_hybrid_ring_layers`)

#### Gated Delta Net

Code: [lcrkv/spec_decode/gdn_spec.py](https://github.com/scatyf3/lcrkv/blob/paper/lcrkv/spec_decode/gdn_spec.py), [lcrkv/kernels/gdn_spec_kernel.py](https://github.com/scatyf3/lcrkv/blob/paper/lcrkv/kernels/gdn_spec_kernel.py)

#### Tree Speculative Decoding

Code: [indep_draft/tree_sd/](https://github.com/scatyf3/IndepDraftEval/tree/main/indep_draft/tree_sd), [indep_draft/sd/tree.py](https://github.com/scatyf3/IndepDraftEval/blob/main/indep_draft/sd/tree.py), [indep_draft/kernels/triton_varlen_tree.py](https://github.com/scatyf3/IndepDraftEval/blob/main/indep_draft/kernels/triton_varlen_tree.py)

</div>
