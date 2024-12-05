<div align="center">
  
# 🚀 NormWear: A Foundation Model for Multivariate Wearable Sensing of Physiological Signals.

[![preprint](https://img.shields.io/static/v1?label=arXiv&message=TBD&color=B31B1B&logo=arXiv)](https://arxiv.org/abs/TBD)
[![huggingface](https://img.shields.io/badge/%F0%9F%A4%97%20Hugging%20Face-Datasets-FFD21E)](https://huggingface.co/datasets/TBD)
[![huggingface](https://img.shields.io/badge/%F0%9F%A4%97%20Hugging%20Face-Models-FFD21E)](https://huggingface.co/collections/TBD)
[![License: MIT](https://img.shields.io/badge/License-Apache--2.0-green.svg)](https://opensource.org/licenses/Apache-2.0)

</div>

## ✨ Introduction
This is the official implementation of the paper [Toward Foundation Model for Multivariate Wearable Sensing of Physiological Signals](https://TBD).

<p align="center">
  <img src="https://drive.google.com/uc?export=view&id=1kLJ9xc7c864mlPzjSo-HCiGRpzdjSMEO" width="100%">
  <br />
</p>

Time-series foundation models have the ability to run inference, mainly forecasting, on any type of time series data, thanks to the informative representations comprising waveform features. Wearable sensing data, on the other hand, contain more variability in both patterns and frequency bands of interest and generally emphasize more on the ability to infer healthcare-related outcomes. The main challenge of crafting a foundation model for wearable sensing physiological signals is to learn generalizable representations that support efficient adaptation across heterogeneous sensing configurations and applications. In this work, we propose NormWear, a step toward such a foundation model, aiming to extract generalized and informative wearable sensing representations. NormWear has been pretrained on a large set of physiological signals, including PPG, ECG, EEG, GSR, and IMU, from various public resources. For a holistic assessment, we perform downstream evaluation on 11 public wearable sensing datasets, spanning 18 applications in the areas of mental health, body state inference, biomarker estimations, and disease risk evaluations. We demonstrate that NormWear achieves a better performance improvement over competitive baselines in general time series foundation modeling. In addition, leveraging a novel representation-alignment-match-based method, we align physiological signals embeddings with text embeddings. This alignment enables our proposed foundation model to perform zero-shot inference, allowing it to generalize to previously unseen wearable signal-based health applications.

## 📈 Usage

Download: TBD

```sh
# Clone the repository
git clone git@github.com:Mobile-Sensing-and-UbiComp-Laboratory/NormWear.git

# Install in editable mode with extra training-related dependencies
cd NormWear && pip install --editable ".[training]"
```

> [!TIP]  
> This repository is intended for research purposes ...

### Extracting Encoder Embeddings

An example showing how to get signal embedding using NormWear:

```python
import pandas as pd  # requires: pip install pandas
import torch
```

### Zero shot inference:

```python
import torch
```

### Linear probing:

```python
import torch
```

## 🔥 Training
TODO

## :floppy_disk: Datasets

TODO

## 📝 Citation

If you find Chronos models useful for your research, please consider citing the associated [paper]([https://arxiv.org/abs/2403.07815](https://TBD)):

```
@misc{tbd,
      title={Toward Foundation Model for Multivariate Wearable Sensing of Physiological Signals}, 
      author={Yunfei Luo, Yuliang Chen, Asif Salekin, Tauhidur Rahman},
      year={2024},
      eprint={tbd},
      archivePrefix={arXiv},
      primaryClass={eess.SP},
      url={https://arxiv.org/abs/tbd}, 
}
```

## marker note

🛡️ 📃
