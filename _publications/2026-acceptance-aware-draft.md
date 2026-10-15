---
title: "Acceptance-Aware Draft Model Training for Speculative Decoding"
collection: publications
permalink: /publication/acceptance-aware-draft
excerpt: 'Training losses that directly target accepted length in speculative decoding: an expected accepted length (EAL) loss for greedy decoding and a window total variation (WTV) loss for sampling, plus an optional GRPO stage rewarded by simulated acceptance length.'
date: 2026-09-21
venue: 'arXiv preprint'
paperurl: 'https://arxiv.org/abs/2609.24150'
citation: 'T. Xia, M. Ganesan, <b>Y. Feng</b>, H. Wang, M. Egger, S. Q. Zhang. "Acceptance-Aware Draft Model Training for Speculative Decoding." <i>arXiv:2609.24150</i>, 2026.'
---

Draft models for speculative decoding are usually trained with cross-entropy or KL-divergence losses, which do not directly optimize the number of tokens accepted per verification step. This work proposes two acceptance-aware objectives: an expected accepted length (EAL) loss for greedy decoding and a window total variation (WTV) loss for sampling-based decoding, along with an optional reinforcement learning stage (GRPO) that uses simulated acceptance length as the reward. Experiments show consistent acceptance-length gains over KL-based training, with WTV strongest under sampling and EAL better suited to greedy verification.
