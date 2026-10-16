---
permalink: /
title: "About Me"
author_profile: true
redirect_from: 
  - /about/
  - /about.html
---

Hi! I'm Yifei Feng, an M.S. student in Computer Engineering at New York University. I work on **efficient LLM inference**, mostly speculative decoding and the serving systems that run it.

👀 **I'm looking for a job or PhD position starting in 2027.** Feel free to reach out!

## Education

<div class="edu-item">
  <img src="/images/logo-nyu.svg" alt="NYU logo">
  <div>
    <strong>New York University</strong><span class="edu-date">2025 – 2027</span><br>
    M.S. in Computer Engineering
  </div>
</div>
<div class="edu-item">
  <img src="/images/logo-bnu.svg" alt="BNU logo">
  <div>
    <strong>Beijing Normal University</strong><span class="edu-date">2021 – 2025</span><br>
    B.S. in Computer Science
  </div>
</div>

## Interests

**Speculative decoding.** A small draft model proposes several tokens and the large target model verifies them in a single forward pass. This cuts memory traffic and exposes more parallelism during decoding. My exploration and insterest are listed as follow:

- **Draft-side KV Cache Pruning(research & engineering):** make each draft step cheaper without significantly hurting acceptance length, achiving the best trade-off between reduce draft overhead and preserving accpetance length.
- **Draft model training**
  - *Formulation (research):* formulate the draft for agreement with the target as EAL Loss, measured by acceptance length, rather than for its own standalone accuracy, thus increase throughput freely by draft finetuning.
  - *Training efficiency (engineering):* speed up and scale draft training with efficient kernels and runtimes.
- **Self-speculative decoding (research):** a separate draft model costs extra memory and training, and never fully agrees with the target. Newer architectures already contain a cheaper version of themselves: an MoE model can draft with fewer experts, and a looped transformer can draft with fewer loop iterations. we can reuse these partial target weights as the draft, so drafting needs no extra model and stays naturally aligned with the target.
- **Speculative decoding in serving engines (research & engineering):** most speculative decoding work is evaluated on a single request, and its gains shrink once requests are batched. I study how to retain part of those gains under batching, and how to schedule speculative decoding better.


**Context compression.** In agentic workloads, every reasoning step and tool call adds to the context, which can grow to hundreds of thousands of tokens over a long task and task gets slow by KV Cache overhead. I look for ways to compress these tokens with minimal accuracy loss.
- **KV cache pruning / sparse attention:** identify redundant KV entries, either training-free or through post-training.
- **Context compaction:** condense long agent trajectories into a compact context, and make the compaction step itself fast.

**Serving engines.** An optimization only matters if people can actually run it. I understand the constraints of state-of-the-art serving engines and designing my optimizations around them. My goal is to make my optimizations can be merged upstream or maintained as a usable fork.

## Get in Touch

I'm always happy to collaborate on research, and glad to chat with students or junior researchers about my work or career path. Email me at [scatyf3@gmail.com](mailto:scatyf3@gmail.com).
