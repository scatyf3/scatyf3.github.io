---
title: "分布式 Speculative Decoding"
title_en: "Distributed Speculative Decoding"
date: 2026-10-17
excerpt: "🚧 Work in progress: splitting draft and target onto separate servers on top of vLLM."
tags:
  - speculative decoding
  - distributed inference
---

<div class="lang-zh" markdown="1">

> 🚧 **施工中。** 这篇还在写，会持续更新。

### motivation

我们定义计算量比访存量为算术强度，衡量此计算在roofline model的哪部分，是计算bound还是mem bound。given这个算数强度，再检查卡的参数，就能找到当前workload在roofline model的哪里。

用这个道理简单计算下，draft和target的workload是非均衡的：

1. draft，$\gamma$步，每一步都是$sP_d$的访存量，$2BP_d$的计算量，算术强度$\approx B$
2. target，一步 $sP_t$的访存量，$2B(\gamma+1)P_t$的计算量，算数强度$\approx B(\gamma+1)$

![Naive scheduling: draft and verify workload imbalance](/images/blog/dsd-naive-scheduling.png)

我们有prefill和decode分离，甚至后续有attention/ffn分离，那有draft和target分离也是非常make sense的，draft side可能要侧重于消除cpu overhead和decoding 特化的优化，部署在hbm带宽好的卡里，target side需要侧重这种小prefill like的优化，部署在算力好点的卡里。

ssd或者sparse spec的paper 也间接优化了这个问题。

1. ssd的本意是消除draft和target的dependancy，但是也间接把他们部署在一个卡上
2. sparse spec非常聪明的利用 long context下的self speculative decoding，修改了调度器，在一张卡上让draft和target parallel，推迟一部分request以换取更好的 throughput，当然这局限于一张卡+self spec的场景，但可能是我今年非常喜欢的一篇mlsys工作

### simulation

从background说回我的任务，很早之前我接触科研，第一个任务就是给[DSDSim](https://arxiv.org/pdf/2511.21669) 这篇paper补end2end的结果，然而当时能力有限+vllm 对自定义修改框架的支持有限，所以并没做出来啥结果。

直到最近继续recycle+准备找工project，当时都甚至准备拉nanovllm魔改了，但调研的时候发现 2026 年 6 月vLLM **v0.22.0** 支持了 自定义轻量drafter，这给我们真正实现paper里写的东西提供了一个简略的路数。

### implementation

我们先讨论最简单的standalone draft with 1 target: N draft，实现overview如下图：

![DSD architecture: draft vLLM server, draft client, router, target vLLM server](/images/blog/dsd-architecture.png)

target 侧：vLLM 原生 `custom_class` 加载 `RemoteDraftProposer`

```python
proposer.configure(client, routing=RoundRobin(), rounds=rounds, window=window, ...)
spec_cfg = {"method": "custom_class",
            "model": "dsdreal.target.proposer.RemoteDraftProposer",   # 类路径
            "num_speculative_tokens": args.gamma}
engine = LLMEngine.from_engine_args(EngineArgs(..., speculative_config=spec_cfg))
```

`RemoteDraftProposer` 内部持有一个 DraftClient，按 `drafter_id` 多路连接 N 个 drafter

接下来就是如何分配 drafter

1. 一个request的不同round是确定性的，每个request确定性link到一个draft server，此后的draft和verify 往返都通向一个server
2. 新来的request会分配draft，用RoundRobin

```python
class RemoteDraftProposer:
    def __init__(self, client, routing):
        self.client = client
        self.routing = routing          # RoundRobin()
        self._seqs = {}                 # seq_id -> _Seq(hist, drafter_id)，跨调用保留

    def _match(self, tokens):
        # given这轮verify的新结果，继续draft
        # tokens是draft的新结果，_seqs存的是之前的结果，通过prefix match，转发给对应的drafter
        for seq in self._seqs.values():
            if tokens[:len(seq.hist)] == seq.hist:
                return seq
        return None

    def propose(self, token_ids_cpu, num_tokens_no_spec):
        requests = []
        for i in range(len(token_ids_cpu)):
            tokens = token_ids_cpu[i][:num_tokens_no_spec[i]]

            seq = self._match(tokens)

            if seq is None: # ① 新请求：选一次 drafter，记到 seq 上
                drafter_id = self.routing.select_drafter(self.client.drafter_ids)
                seq = _Seq(seq_id=new_id(), drafter_id=drafter_id, hist=[])
                self._seqs[seq.seq_id] = seq
                prompt, delta = tokens, []
            else:  # ② 老请求：不碰 drafter_id
                prompt, delta = None, tokens[len(seq.hist):]

            seq.hist = tokens
            requests.append(SpeculateRequest(
                seq_id=seq.seq_id,
                drafter_id=seq.drafter_id,
                prompt_token_ids=prompt,
                accepted_tokens=delta,
            ))
        return self.client.speculate(requests)
```

截止到现在，draft client里维护的只有naive的 通过token prefix matching seq_id → draft的映射，怎么evict掉一个完整的request？

1. 每次 `propose()` 都会更新本步出现的序列的 `last_seen`
2. 通过超时判断draft结束，每 256 次调用跑一次 `_gc()`，清理并发 `free`

```python
class RemoteDraftProposer:
    STALE_AFTER_CALLS = 2000            # 这么多次 propose() 没出现 → 认为请求结束了

    def __init__(self, client):
        self.client = client
        self._seqs = {}                 # seq_id -> _Seq(..., last_seen)
        self._calls = 0                 # propose() 被调用的次数，当作"时钟"

    def propose(self, token_ids_cpu, num_tokens_no_spec):
        self._calls += 1                # 时钟 +1
        for tokens in 本步每个槽位的 token 序列:
            seq = 按前缀匹配找到 seq，找不到就新建并选 drafter
            ...                          # 组装 SpeculateRequest
            seq.last_seen = self._calls  # ① 本步出现了，刷新 last_seen
        if self._calls % 256 == 0:       # ② 每 256 次调用扫一遍
            self._gc()
        return 发给 drafter 并返回草稿

    def _gc(self):
        stale = [sid for sid, s in self._seqs.items()
                 if self._calls - s.last_seen > self.STALE_AFTER_CALLS]
        if stale:
            for sid in stale:
                del self._seqs[sid]  # 本地映射表里删掉
            self.client.free(stale)  # 通知所有 drafter 释放这些 seq 的状态
```

介绍完毕了target side的状态管理，接下来我们介绍如何通信

1. prefill：传prompt
2. decode：只传 `accepted_tokens` 的 token id + seq 元数据
   1. target to draft：seq_id + gamma + accept token + num_confirmed
   2. draft to target：seq_id + draft_token + draft_ms

draft server：ZMQ/HTTP 包一个离线 vLLM 引擎，对已确认前缀 `generate(max_tokens=γ)`，靠 prefix cache 复用 KV，batching计算当前收到的全部请求，返回 γ 个 token

### limitation

1. 并非paper里说的「draft driven」，事实上由于vllm的sd调度策略，一定是target driven。除非深入改调度器，然而超出了学生个人project和simulator paper的scope

</div>

<div class="lang-en" markdown="1">

> 🚧 **Work in progress.** This post is still being written and will keep getting updated.

### Motivation

Arithmetic intensity is compute divided by memory traffic. It tells you where a computation sits on the roofline model: compute bound or memory bound. Given the arithmetic intensity and the GPU's specs, you can place the current workload on the roofline.

Applying this to speculative decoding shows that the draft and target workloads are unbalanced:

1. Draft: $\gamma$ steps, each with $sP_d$ memory traffic and $2BP_d$ FLOPs, so arithmetic intensity $\approx B$.
2. Target: one step with $sP_t$ memory traffic and $2B(\gamma+1)P_t$ FLOPs, so arithmetic intensity $\approx B(\gamma+1)$.

![Naive scheduling: draft and verify workload imbalance](/images/blog/dsd-naive-scheduling.png)

We already disaggregate prefill and decode, and more recently attention and FFN, so disaggregating draft and target makes a lot of sense. The draft side cares about removing CPU overhead and decode-specific optimizations, and belongs on GPUs with high HBM bandwidth. The target side looks like a small prefill, and belongs on GPUs with more compute.

The SSD and sparse-spec papers address this problem indirectly:

1. SSD's goal is to remove the dependency between draft and target, but along the way it also places them on the same GPU.
2. Sparse spec cleverly uses self-speculative decoding for long context. It modifies the scheduler so draft and target run in parallel on one GPU, delaying some requests in exchange for better throughput. It is limited to one GPU and self-speculation, but it is probably my favorite MLSys paper this year.

### Simulation

Back to my own task. When I first started doing research, my first job was to add end-to-end results to the [DSDSim](https://arxiv.org/pdf/2511.21669) paper. My skills were limited then, and vLLM had little support for custom modifications, so I didn't get much out of it.

Recently I came back to it as a job-search project. I was about to fork nano-vllm and hack on it, but while surveying I found that vLLM **v0.22.0** (June 2026) supports custom lightweight drafters. That gives a simple path to actually building what the paper describes.

### Implementation

Start with the simplest setup: standalone drafts with 1 target : N drafts. The overview:

![DSD architecture: draft vLLM server, draft client, router, target vLLM server](/images/blog/dsd-architecture.png)

On the target side, vLLM's native `custom_class` loads `RemoteDraftProposer`:

```python
proposer.configure(client, routing=RoundRobin(), rounds=rounds, window=window, ...)
spec_cfg = {"method": "custom_class",
            "model": "dsdreal.target.proposer.RemoteDraftProposer",   # class path
            "num_speculative_tokens": args.gamma}
engine = LLMEngine.from_engine_args(EngineArgs(..., speculative_config=spec_cfg))
```

`RemoteDraftProposer` holds a DraftClient, which multiplexes connections to N drafters by `drafter_id`.

Next is how drafters are assigned:

1. The rounds of a request are deterministic: each request is pinned to one draft server, and every later draft/verify round trip goes to that server.
2. New requests are assigned a drafter round-robin.

```python
class RemoteDraftProposer:
    def __init__(self, client, routing):
        self.client = client
        self.routing = routing          # RoundRobin()
        self._seqs = {}                 # seq_id -> _Seq(hist, drafter_id), kept across calls

    def _match(self, tokens):
        # given this round's verify result, keep drafting:
        # match the new tokens against stored history by prefix and route to that drafter
        for seq in self._seqs.values():
            if tokens[:len(seq.hist)] == seq.hist:
                return seq
        return None

    def propose(self, token_ids_cpu, num_tokens_no_spec):
        requests = []
        for i in range(len(token_ids_cpu)):
            tokens = token_ids_cpu[i][:num_tokens_no_spec[i]]

            seq = self._match(tokens)

            if seq is None: # ① new request: pick a drafter once and store it on seq
                drafter_id = self.routing.select_drafter(self.client.drafter_ids)
                seq = _Seq(seq_id=new_id(), drafter_id=drafter_id, hist=[])
                self._seqs[seq.seq_id] = seq
                prompt, delta = tokens, []
            else:  # ② existing request: keep its drafter_id
                prompt, delta = None, tokens[len(seq.hist):]

            seq.hist = tokens
            requests.append(SpeculateRequest(
                seq_id=seq.seq_id,
                drafter_id=seq.drafter_id,
                prompt_token_ids=prompt,
                accepted_tokens=delta,
            ))
        return self.client.speculate(requests)
```

So far the draft client only keeps a naive seq_id → drafter map built by token prefix matching. How do we evict a finished request?

1. Every `propose()` call updates `last_seen` for the sequences that appear in that step.
2. A request is considered finished after a timeout: every 256 calls, `_gc()` cleans up stale entries and sends `free`.

```python
class RemoteDraftProposer:
    STALE_AFTER_CALLS = 2000            # unseen for this many propose() calls -> request is done

    def __init__(self, client):
        self.client = client
        self._seqs = {}                 # seq_id -> _Seq(..., last_seen)
        self._calls = 0                 # number of propose() calls, used as a clock

    def propose(self, token_ids_cpu, num_tokens_no_spec):
        self._calls += 1                # tick
        for tokens in token_sequences_in_this_step:
            seq = match_by_prefix_or_create_and_pick_drafter(tokens)
            ...                          # build SpeculateRequest
            seq.last_seen = self._calls  # ① seen this step, refresh last_seen
        if self._calls % 256 == 0:       # ② sweep every 256 calls
            self._gc()
        return send_to_drafters_and_return_drafts()

    def _gc(self):
        stale = [sid for sid, s in self._seqs.items()
                 if self._calls - s.last_seen > self.STALE_AFTER_CALLS]
        if stale:
            for sid in stale:
                del self._seqs[sid]  # drop from the local map
            self.client.free(stale)  # tell every drafter to release these seqs
```

That covers state management on the target side. Next, communication:

1. Prefill: send the prompt.
2. Decode: send only the token ids of `accepted_tokens` plus sequence metadata.
   1. Target to draft: seq_id + gamma + accepted tokens + num_confirmed
   2. Draft to target: seq_id + draft tokens + draft_ms

Draft server: an offline vLLM engine wrapped in ZMQ/HTTP. It runs `generate(max_tokens=γ)` on the confirmed prefix, reuses KV through the prefix cache, batches all requests it has received, and returns γ tokens.

### Limitations

1. This is not the "draft driven" design described in the paper. Because of vLLM's speculative decoding scheduler, it is always target driven. Changing that means going deep into the scheduler, which is beyond the scope of a personal student project and a simulator paper.

</div>
