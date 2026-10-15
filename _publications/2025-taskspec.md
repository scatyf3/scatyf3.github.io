---
title: "Automatic Task Detection and Heterogeneous LLM Speculative Decoding"
collection: publications
permalink: /publication/taskspec
excerpt: 'Automatically partitions downstream tasks and routes prompts to task-aligned heterogeneous draft models: 6–50% higher draft accuracy and 1.10–2.64× speedup.'
date: 2025-05-13
venue: 'arXiv preprint'
paperurl: 'https://arxiv.org/abs/2505.08600'
citation: 'D. Ge, J. Gao, Q. Jiang, <b>Y. Feng</b>, W. Ji. "Automatic Task Detection and Heterogeneous LLM Speculative Decoding." <i>arXiv:2505.08600</i>, 2025.'
---

Speculative decoding, which combines a draft model with a target model, has emerged as an effective approach to accelerate large language model (LLM) inference. However, existing methods often face a trade-off between the acceptance rate and decoding speed in downstream tasks due to the limited capacity of the draft model, making it difficult to ensure efficiency across diverse tasks. To address this problem, we propose a speculative decoding algorithm tailored for downstream task optimization. It includes an automatic task partitioning and assigning method, which automatically categorizes downstream tasks into different sub-tasks and assigns them to a set of heterogeneous draft models. Each draft model is aligned with the target model using task-specific data, thereby enhancing the consistency of inference results. In addition, our proposed method incorporates an online lightweight prompt classifier to dynamically route prompts to the appropriate draft model. Experimental results demonstrate that the proposed method improves draft accuracy by 6% to 50% over vanilla speculative decoding, while achieving a speedup of 1.10x to 2.64x in LLM inference.