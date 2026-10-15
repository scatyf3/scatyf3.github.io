---
title: "GSR-GNN: Training Acceleration and Memory-Saving Framework of Deep GNNs on Circuit Graph"
collection: publications
permalink: /publication/gsrgnn
excerpt: 'Reversible residual modules plus group-wise sparse non-linear operators for training deep GNNs on circuit graphs: up to 87.2% memory savings and over 30× speedup.'
date: 2026-03-27
venue: 'DAC 2026'
paperurl: 'https://arxiv.org/abs/2603.27156'
citation: 'Y. Luo, S. Li, <b>Y. Feng</b>, et al. "GSR-GNN: Training Acceleration and Memory-Saving Framework of Deep GNNs on Circuit Graph." <i>DAC 2026</i>.'
---

Graph Neural Networks (GNNs) have shown great potential in analyzing circuit graphs, compared to traditional analytical methods. However, with the increasing scale and complexity of circuit graphs, conventional GNNs with several layers are overstretched. We observe that training deep GNNs can lead to excellent performance on circuit graphs; however, current deep GNN training frameworks face significant runtime and memory overhead issues on GPUs. In this paper, we propose Grouped-Sparse-Reversible GNN (GSR-GNN), a highly computationally- and memory-efficient, domain-specific training acceleration framework. It utilizes reversible residual modules to support the training of deep GNNs up to hundreds of layers with a specialized, group-wise, sparse non-linear operator to realize effective embedding compression. The optimized workflow resolves fragmented and computationally burdensome variable management. Our framework strikes a feasible balance between memory savings and training speed on GPUs without compromising model performance. Our evaluation results on sampled circuit graphs show that our framework can reach up to 87.2% memory savings, and over 30× acceleration without a major impact on model correlation metrics.
