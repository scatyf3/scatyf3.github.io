---
title: "SpecCompact: Accelerating LLM Agent Context Compaction with Slot-Parallel Semantic Speculative Verification"
collection: publications
permalink: /publication/speccompact
excerpt: 'A small drafter generates structured summary slots in parallel and the target model verifies and repairs them, giving 2.04–5.63× context-compaction speedups for LLM agents.'
date: 2026-11-04
venue: 'arXiv preprint'
paperurl: 'https://arxiv.org/abs/2611.06876'
citation: 'Y. An, <b>Y. Feng</b>, X. Yin. "SpecCompact: Accelerating LLM Agent Context Compaction with Slot-Parallel Semantic Speculative Verification." <i>arXiv:2611.06876</i>, 2026.'
---

LLM agents use context compaction to manage growing interaction histories, but summarizing long histories with large models adds noticeable latency. We observe that structured summary slots can retain information when generated independently. SpecCompact fine-tunes a small drafter to generate slots in parallel; the target model verifies each proposed line and repairs rejected content, and system optimizations reduce repeated context processing and model calls. Across AppWorld and τ³ benchmarks with two target models, SpecCompact achieves 2.04–5.63× compaction speedups over Hermes-style summarization. Downstream evaluations show model- and task-dependent performance trade-offs, and a user study within Hermes shows an 8.68× median compaction speedup.
