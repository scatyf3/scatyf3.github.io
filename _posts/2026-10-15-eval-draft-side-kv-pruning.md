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

HF 对 SWA 有三种 cache 实现：(1) `DynamicSlidingWindowLayer`，每步 cat 后截断到最近 window−1 个 token，shape 动态，且被逐出的 KV 无法回滚；(2) `StaticSlidingWindowLayer`，固定 window 大小的 buffer，写满后每步 `roll` 整体平移；(3) 保留全部 KV，用 sliding mask 屏蔽窗口外的位置。三者都不是 ring buffer。

```python
# ① DynamicSlidingWindowLayer
full = torch.cat([self.keys, key_states], dim=-2)
self.keys = full[:, :, -self.sliding_window + 1:, :]       # 只留最后 window-1 个, 窗口外的直接丢掉

# ② StaticSlidingWindowLayer
if not is_full:
    pos = torch.arange(q_len) + self.cumulative_length
    self.keys.index_copy_(2, pos, key_states)
elif q_len == 1:   # 已满 + decode: 整体左移一格, 再写最后一个槽位
    new_keys = self.keys.roll(-1, dims=-2)
    new_keys[:, :, [-1]] = key_states
    self.keys.copy_(new_keys) # copy_ 回原地址, 保证地址不变
else: # 已满 + 多个 token: cat 后截断
    full = torch.cat([self.keys[:, :, 1:], key_states], dim=-2)
    self.keys.copy_(full[:, :, -window:])

# ③ 全部 KV + sliding mask
self.keys = torch.cat([self.keys, key_states], dim=-2)
# 什么都不丢
# 窗口由 mask 实现 (masking_utils.sliding_window_causal_mask_function):
mask = (kv_idx <= q_idx) & (kv_idx > q_idx - sliding_window)
```

但这里有两个问题：

1. **SD 兼容性**：除了 sliding mask 之外的方法都不兼容 SD，因为走出窗口的 KV 会被永久 evict。我们需要一个额外的 KV buffer，保留「已经离开 window，但可能因为 draft rollback 回到 window」的 KV。
2. **Graph capture**：① 无法被 capture；② 可以，但每次 decode 都要移动整个窗口，感觉性能上不太好（不过这部分没测）。

Given 这些问题，我们还是要自己造一个兼容 SD、可以被 capture 的 SWA buffer。先讲下已有框架的做法：

1. **vLLM**：paged KV 天然适配真 SWA，直接 head + tail 加 mask，然后 evict head page、append tail page 即可。
2. **SGLang** 的 radix cache 有两个选择：
    1. 同 paged attention；
    2. 一片连续的 SWA 区域，用 ring buffer，**DeepSeek-V4 和 DSpark draft 用的是这种**。

很显然，出于 research demo 的动机，我们没必要搞 paged 这么复杂，直接在 single request 上用 ring buffer 比较好，或者说 page 就是更粗粒度的 ring buffer。

**What is ring buffer?**

1. 我们 preallocate size = W 的 KV cache；
2. KV 写入的 address = pos % W。

**Why ring buffer?** 位置编码只 care 相对距离，所以 KV pair 不严格按照顺序存放，不影响 output。

数学推导：

**记号**：query 的位置是 m，第 j 个 key/value 的逻辑位置是 $p_j$，存放在 slot $\sigma(j)$。ring buffer 就是 $\sigma(j) = p_j \bmod W$。

**① 单个 query 的 attention 输出**

$$o_m=\sum_{j\in\mathcal{K}}\frac{\exp(s_{mj})}{\sum_{l\in\mathcal{K}}\exp(s_{ml})}\,v_j, \qquad s_{mj}=\frac{\tilde q_m^\top \tilde k_j}{\sqrt d}+b_{mj}$$

其中 $\mathcal{K}$ 是这个 query 能看到的 key 的集合，$$b_{mj}\in\{0,-\infty\}$$ 是 mask。

**② 存放顺序不影响结果**

对 slot 做任意重排 $\pi$，分子和分母都是对同一个集合 $\mathcal{K}$ 求和，加法满足交换律，所以

$$o_m=\sum_{j}\alpha_{mj}v_j=\sum_{j}\alpha_{m\pi(j)}v_{\pi(j)}.$$

前提是每个 $s_{mj}$ 只依赖 token 本身，也就是只依赖 $(q_m, k_j, p_j, m)$，**不依赖它存放在哪个 slot**。下面两步说明这个前提成立。

**③ RoPE 的相位在写入时就固定了**

RoPE 对 query 和 key 分别乘上各自位置的旋转矩阵：$\tilde q_m = R_m q_m$，$\tilde k_j = R_{p_j}k_j$。旋转矩阵是正交的，满足 $R_a^\top R_b = R_{b-a}$，所以

$$\tilde q_m^\top \tilde k_j = q_m^\top R_m^\top R_{p_j} k_j = q_m^\top R_{p_j-m}\,k_j .$$

这个分数只依赖相对距离 $p_j - m$。而且 $R_{p_j}$ 是在写进 cache **之前**、按真实位置 $p_j$ 乘上去的，后面不管这个 K 存到哪个 slot，它的相位都不会变。

**④ mask 也必须按逻辑位置计算**

sliding window 加 causal 的 mask 是

$$b_{mj}=\begin{cases}0,& m-W< p_j\le m\\-\infty,&\text{otherwise}\end{cases}$$

它只依赖 $p_j$ 和 m。如果 kernel 按 slot 编号去判断窗口（比如假设 slot 是连续递增的），ring 一回绕结果就错了。所以 kernel 要么拿到每个 slot 的 `slot_pos`，要么在回绕点把 buffer 切成两段连续区间分别计算，再用 LSE 合并。

另外，当窗口已经写满、并且是 decode（Q=1）时，buffer 里的每个 slot 都满足 $$m-W<p_j\le m$$，mask 可以全部放开，这时什么都不用传。

**⑤ 结论**

只要 ③ 和 ④ 都成立，$s_{mj}$ 就只依赖 token 本身，不依赖 slot。再由 ②，按 $p_j \bmod W$ 存放和按顺序存放，得到的 $o_m$ 数学上完全相同。

**Ring buffer + draft？** 在 ring buffer 之外加一个 size = γ 的 draft scratch，按 verify 通过的 commit token 推入 draft ring buffer 即可。

Code：[lcrkv/kv_cache.py](https://github.com/scatyf3/lcrkv/blob/paper/lcrkv/kv_cache.py)（`RingScratchLayer`、`force_hybrid_ring_layers`）

#### Gated Delta Net

我们还 eval 了 Qwen3.5 上的效果：保持 GDN 层不被 prune，只对 full attention 层的 KV 做 pruning。

**Kernel**：这里需要一个 GDN verify kernel。TL;DR：HF 提供的 GDN kernel 只有 prefill 和 decode 两种，prefill 默认从 0 开始，会静默忽略已有的 recurrent state，所以没有 SD verify 风格的 kernel。我们从 vLLM 借鉴了一个。

**Runtime**：recurrent state 需要状态回滚管理，无法像 KV 一样简单地 crop 掉不要的部分。做法也很简单：

1. 对 draft：维护 γ+1 个 recurrent state 的 slot，每步 decode 都保留，verify 之后回退到对应的 slot；
2. 对 target 同理：我们的 verify kernel 能产出 per-token 的 recurrent state，也同样保留。

Code：[lcrkv/spec_decode/gdn_spec.py](https://github.com/scatyf3/lcrkv/blob/paper/lcrkv/spec_decode/gdn_spec.py)、[lcrkv/kernels/gdn_spec_kernel.py](https://github.com/scatyf3/lcrkv/blob/paper/lcrkv/kernels/gdn_spec_kernel.py)

#### Tree Speculative Decoding

<!-- TODO: recap tree SD 的流程 -->

**SDPA 的坑**

接下来的目标是把 evaluation extend 到 tree，但在 tree kernel 这里又卡住了。我们用 torch SDPA 对比 chain 和 tree 时 SD 的 time breakdown，在 GSM8K 这种短上下文的数据上发现 tree 的 draft time 比 chain 莫名其妙多了 10ms。多出来的时间集中在 attention kernel 而非 e2e，所以 CUDA Graph 也无法优化这个问题。

这里一度卡住很久。后来因为做其他的事情，要用 nsys profile memory traffic，才发现 SDPA + tree mask 时，SDPA 为了正确性会自动 fallback 到最 naive 的 GQA，直接 expand KV，导致 attention kernel 的 memory traffic 比 chain 多了一倍。

![Hugging Face use_gqa_in_sdpa: a non-None attention mask falls back to repeat_kv](/images/blog/sdpa-mask-repeat-kv.png)
*图 4：attention mask 非 None 时，直接 repeat KV。*

**手写 tree kernel**

SDPA 有一堆问题，所以还是仿照 chain 手写一个读 valid 长度的 kernel。然而 tree 比起 chain 又带来若干问题：

1. **Dynamic Q**：tree 每层的宽度不一样，Q 的 `seq_len` 可能为 1、2、4、8，最大为 beam search 的 top-k；
2. **Dynamic tree**：在 beam search 风格的 EAGLE-2 里，我们不预先定义 tree 的 shape，而是按接受置信度动态扩展 top-k 个 node。

**Dynamic Q**：按 Q 分桶 capture CUDA Graph，capture {1, 2, K} 三个 graph，每个 graph 有各自**固定地址的输入 buffer**。三个 shape 分别对应：

- Q=1：root；
- Q=2：root 加上上一步接收但未 forward 的 bonus / repair token，一次 forward 补上它缺的 K/V；
- Q=K：之后 S−1 层 beam forward 全用这个。

**Dynamic tree**：KV preallocate 和 dynamic Q 的问题前面已经提过，剩下的是 mask。naive 的 dynamic tree 每步在 CPU 上随着 tree 的 shape 重新构建 mask（参考 EAGLE-2 的 codebase），导致每次 tree forward 都要 recapture。我们的做法是 preallocate 一个 static mask，每步只根据置信度用 torch kernel 修改 mask，同时给 mask 也配一个记录 valid position 的 tensor。

<!-- TODO: 这里的写法 -->

在 target side，我们还是用 SDPA 做 verify：首先对略大的模型，launch bound 没那么严重；其次这是一个 research demo，没必要追求极致性能，保持灵活更好。不过我们手动 patch 掉了那个低效的 GQA repeat KV，保证基本的效率。

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

Hugging Face has three cache implementations for SWA: (1) `DynamicSlidingWindowLayer`, which concatenates every step and then truncates to the most recent window−1 tokens; its shape is dynamic and evicted KV cannot be rolled back. (2) `StaticSlidingWindowLayer`, a fixed window-sized buffer that, once full, shifts everything with `roll` every step. (3) Keep all KV and hide positions outside the window with a sliding mask. None of the three is a ring buffer.

```python
# ① DynamicSlidingWindowLayer
full = torch.cat([self.keys, key_states], dim=-2)
self.keys = full[:, :, -self.sliding_window + 1:, :]       # keep only the last window-1; anything outside the window is dropped

# ② StaticSlidingWindowLayer
if not is_full:
    pos = torch.arange(q_len) + self.cumulative_length
    self.keys.index_copy_(2, pos, key_states)
elif q_len == 1:   # full + decode: shift everything left by one, then write the last slot
    new_keys = self.keys.roll(-1, dims=-2)
    new_keys[:, :, [-1]] = key_states
    self.keys.copy_(new_keys) # copy_ back in place so the address stays fixed
else: # full + multiple tokens: concatenate, then truncate
    full = torch.cat([self.keys[:, :, 1:], key_states], dim=-2)
    self.keys.copy_(full[:, :, -window:])

# ③ all KV + sliding mask
self.keys = torch.cat([self.keys, key_states], dim=-2)
# nothing is dropped
# the window comes from the mask (masking_utils.sliding_window_causal_mask_function):
mask = (kv_idx <= q_idx) & (kv_idx > q_idx - sliding_window)
```

This leaves two problems:

1. **SD compatibility**: every option except the sliding mask is incompatible with SD, because KV that leaves the window is evicted for good. We need an extra KV buffer to hold KV that has left the window but may come back into it after a draft rollback.
2. **Graph capture**: ① cannot be captured. ② can, but it shifts the whole window on every decode step, which seems bad for performance (I didn't measure this, though).

Given these problems, we still had to build our own SWA buffer that works with SD and can be captured. First, how existing frameworks do it:

1. **vLLM**: paged KV fits true SWA naturally. Mask the head and tail, then evict the head page and append a tail page.
2. **SGLang**'s radix cache has two options:
    1. The same as paged attention;
    2. One contiguous SWA region managed as a ring buffer. **DeepSeek-V4 and the DSpark draft use this.**

Since this is a research demo, there's no need for anything as complex as paging. A ring buffer on a single request is the better fit; a page is really just a coarser-grained ring buffer.

**What is a ring buffer?**

1. Preallocate a KV cache of size W;
2. Write KV at address = pos % W.

**Why a ring buffer?** Positional encoding only cares about relative distance, so storing KV pairs out of order does not change the output.

The math:

**Notation**: the query is at position m; the j-th key/value has logical position $p_j$ and is stored in slot $\sigma(j)$. A ring buffer is simply $\sigma(j) = p_j \bmod W$.

**① Attention output for one query**

$$o_m=\sum_{j\in\mathcal{K}}\frac{\exp(s_{mj})}{\sum_{l\in\mathcal{K}}\exp(s_{ml})}\,v_j, \qquad s_{mj}=\frac{\tilde q_m^\top \tilde k_j}{\sqrt d}+b_{mj}$$

where $\mathcal{K}$ is the set of keys this query can see and $$b_{mj}\in\{0,-\infty\}$$ is the mask.

**② Storage order does not affect the result**

Apply any permutation $\pi$ to the slots. Numerator and denominator both sum over the same set $\mathcal{K}$, and addition is commutative, so

$$o_m=\sum_{j}\alpha_{mj}v_j=\sum_{j}\alpha_{m\pi(j)}v_{\pi(j)}.$$

This assumes each $s_{mj}$ depends only on the token itself, i.e. only on $(q_m, k_j, p_j, m)$, and **not on which slot it is stored in**. The next two steps show this holds.

**③ RoPE's phase is fixed at write time**

RoPE multiplies the query and key by the rotation matrix for their own positions: $\tilde q_m = R_m q_m$, $\tilde k_j = R_{p_j}k_j$. Rotation matrices are orthogonal and satisfy $R_a^\top R_b = R_{b-a}$, so

$$\tilde q_m^\top \tilde k_j = q_m^\top R_m^\top R_{p_j} k_j = q_m^\top R_{p_j-m}\,k_j .$$

The score depends only on the relative distance $p_j - m$. And $R_{p_j}$ is applied with the true position $p_j$ **before** the key is written to the cache, so its phase doesn't change no matter which slot the K ends up in.

**④ The mask must also use logical positions**

The sliding-window plus causal mask is

$$b_{mj}=\begin{cases}0,& m-W< p_j\le m\\-\infty,&\text{otherwise}\end{cases}$$

It depends only on $p_j$ and m. If the kernel decides the window from slot indices (for example, assuming slots increase contiguously), the result is wrong as soon as the ring wraps around. So the kernel must either get each slot's `slot_pos`, or split the buffer at the wrap point into two contiguous ranges, compute each, and merge them with LSE.

Also, once the window is full and we are decoding (Q=1), every slot in the buffer satisfies $$m-W<p_j\le m$$, so the mask can be dropped entirely and nothing needs to be passed in.

**⑤ Conclusion**

As long as ③ and ④ hold, $s_{mj}$ depends only on the token, not the slot. Then by ②, storing at $p_j \bmod W$ and storing in order give mathematically identical $o_m$.

**Ring buffer + draft?** Add a draft scratch of size γ next to the ring buffer, and push the tokens committed by verify into the draft ring buffer.

Code: [lcrkv/kv_cache.py](https://github.com/scatyf3/lcrkv/blob/paper/lcrkv/kv_cache.py) (`RingScratchLayer`, `force_hybrid_ring_layers`)

#### Gated Delta Net

We also evaluated on Qwen3.5: GDN layers are left unpruned, and only the full-attention layers' KV is pruned.

**Kernel**: this needs a GDN verify kernel. TL;DR: the GDN kernels in Hugging Face only cover prefill and decode, and prefill always starts from 0, silently ignoring any existing recurrent state, so there is no SD-verify-style kernel. We borrowed one from vLLM.

**Runtime**: the recurrent state needs rollback management; you can't just crop off the unwanted part as with KV. The fix is simple:

1. Draft: keep γ+1 recurrent-state slots, save one at every decode step, and roll back to the matching slot after verify;
2. Target: same idea. Our verify kernel produces a per-token recurrent state, which we keep the same way.

Code: [lcrkv/spec_decode/gdn_spec.py](https://github.com/scatyf3/lcrkv/blob/paper/lcrkv/spec_decode/gdn_spec.py), [lcrkv/kernels/gdn_spec_kernel.py](https://github.com/scatyf3/lcrkv/blob/paper/lcrkv/kernels/gdn_spec_kernel.py)

#### Tree Speculative Decoding

<!-- TODO: recap the tree SD procedure -->

**The SDPA trap**

The next goal was to extend the evaluation to trees, but we got stuck again on the tree kernel. Comparing the SD time breakdown for chain vs. tree with torch SDPA, we found on short-context data such as GSM8K that the tree's draft time was inexplicably 10 ms longer than the chain's. The extra time was in the attention kernel rather than end to end, so CUDA Graph couldn't help.

This blocked us for a long time. Later, while profiling memory traffic with nsys for something else, I found that with SDPA + a tree mask, SDPA falls back to the most naive GQA path for correctness and expands the KV directly, so the attention kernel's memory traffic is double that of the chain case.

![Hugging Face use_gqa_in_sdpa: a non-None attention mask falls back to repeat_kv](/images/blog/sdpa-mask-repeat-kv.png)
*Figure 4: when the attention mask is not None, KV is simply repeated.*

**A hand-written tree kernel**

SDPA has a pile of problems, so as with the chain, we hand-wrote a kernel that reads the valid length. But trees bring several new problems compared with chains:

1. **Dynamic Q**: each tree layer has a different width, so Q's `seq_len` can be 1, 2, 4 or 8, up to the beam search top-k;
2. **Dynamic tree**: in beam-search-style EAGLE-2, the tree's shape is not predefined; it expands the top-k nodes by acceptance confidence on the fly.

**Dynamic Q**: capture CUDA Graphs bucketed by Q: three graphs for {1, 2, K}, each with its own **fixed-address input buffer**. The three shapes are for:

- Q=1: the root;
- Q=2: the root plus the bonus / repair token accepted in the previous step but not yet forwarded, filling in its missing K/V in one forward;
- Q=K: all of the following S−1 beam forward layers.

**Dynamic tree**: KV preallocation and dynamic Q are covered above; what remains is the mask. A naive dynamic tree rebuilds the mask on the CPU each step as the tree's shape changes (see the EAGLE-2 codebase), which forces a recapture on every tree forward. Instead, we preallocate a static mask, update it each step with a torch kernel based only on the confidences, and give the mask its own valid-position tensor.

<!-- TODO: how this is written -->

On the target side we still use SDPA for verify. First, launch bound is less severe for the larger model; second, this is a research demo, so there's no need for peak performance, and staying flexible is better. We did manually patch out the inefficient GQA repeat KV to keep basic efficiency.

Code: [indep_draft/tree_sd/](https://github.com/scatyf3/IndepDraftEval/tree/main/indep_draft/tree_sd), [indep_draft/sd/tree.py](https://github.com/scatyf3/IndepDraftEval/blob/main/indep_draft/sd/tree.py), [indep_draft/kernels/triton_varlen_tree.py](https://github.com/scatyf3/IndepDraftEval/blob/main/indep_draft/kernels/triton_varlen_tree.py)

</div>
