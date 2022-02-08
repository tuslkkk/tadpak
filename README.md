# Towards a Rigorous Evaluation of Time-series Anomaly Detection (AAAI 2022)
The implementation of PA%K evaluation protocol of Towards a Rigorous Evaluation of Time-sereis anomaly detection, Siwon Kim, Kukjin Choi, Hyun-Soo Choi, Byunghan Lee, and Sungroh Yoon, AAAI 2022. [paper](https://arxiv.org/abs/2109.05257)

# Abstract
---
In recent years, proposed studies on time-series anomaly detection (TAD) report high F1 scores on benchmark TAD datasets, giving the impression of clear improvements. However, most studies apply a peculiar evaluation protocol called point adjustment (PA) before scoring. In this paper, we theoretically and experimentally reveal that the PA protocol has a great possibility of overestimating the detection performance; that is, even a random anomaly score can easily turn into a state-of-the-art TAD method. Therefore, the comparison of TAD methods with F1 scores after the PA protocol can lead to misguided rankings. Furthermore, we question the potential of existing TAD methods by showing that an untrained model obtains comparable detection performance to the existing methods even without PA. Based on our findings, we propose a new baseline and an evaluation protocol. We expect that our study will help a rigorous evaluation of TAD and lead to further improvement in future researches.

# Installation
---
```python
pip install tadpak
```

#Usage
---
1. Adjust prediction with PA%K protocol
```python
from tadpak import pak

# scores  : predicted anomaly score
# targets : ground truth labels
# thres   : anomaly score threshold
# k       : pak threshold

adjusted_preds = pak.pak(scores, targets, thres, k=20)
```

2. Evaluate predictions
```python
import tadpak import evaluate

results = evaluate.evaluate(scores, targets)
```
