# RAIL: Region-Aware Instructive Learning for Semi-Supervised 3D Tooth Segmentation from CBCT

by Chuyu Zhao, Hao Huang, Jiashuo Guo, Ziyu Shen, Jie Liu 

![Python Version](https://img.shields.io/badge/Python-3.8%2B-blue) 
![License](https://img.shields.io/github/license/Tournesol-Saturday/RAIL) 
![Last Commit](https://img.shields.io/github/last-commit/Tournesol-Saturday/RAIL) 
![Stars](https://img.shields.io/github/stars/Tournesol-Saturday/RAIL?style=social) 
![Issues](https://img.shields.io/github/issues/Tournesol-Saturday/RAIL) 
![Forks](https://img.shields.io/github/forks/Tournesol-Saturday/RAIL)

RAIL is a semi-supervised learning framework designed to improve the accuracy and robustness of 3D tooth segmentation from Cone-Beam Computed Tomography (CBCT) scans, particularly when labeled data is scarce. The framework leverages region-aware instruction mechanisms to focus on anatomically complex or ambiguous areas, significantly enhancing segmentation performance in the presence of unreliable pseudo-labels.

## News!

- **[04/14/2025]** RAIL framework code and models are now available! Please check out the [GitHub repository](#) for more details.

## Overview

RAIL (Region-Aware Instructive Learning) is a novel dual-group, dual-student semi-supervised framework for medical image segmentation, specifically designed to address the issue of limited labeled data in CBCT-based 3D tooth segmentation. This approach introduces two key mechanisms:
- **Disagreement-Focused Supervision (DFS) Controller:** Focuses on areas where model predictions diverge to improve supervision in anatomically ambiguous or mislabeled regions.
- **Confidence-Aware Learning (CAL) Modulator:** Enhances model stability by reinforcing high-confidence predictions and suppressing low-confidence pseudo-labels.

RAIL outperforms state-of-the-art methods under limited annotation scenarios, improving both segmentation accuracy and model reliability in medical imaging tasks.

## Installation

To perform inference locally with the debugger GUI, follow these steps:

```bash
git clone https://github.com/Tournesol-Saturday/RAIL.git;
cd ./RAIL;
pip install -r requirements.txt
```

Download the model checkpoint and save it at `./models/model.pt`.

## Dataset



## Usage

```
# after install dependcies
git clone git@github.com:Axi404/PMT.git
cd PMT/code
python train_PMT.py
python test_LA.py
```

## Reference

If you use this project in your work, please cite the following paper:

```
```



## Acknowledgements

We would like to acknowledge the contributions of the following projects and datasets:

- [PMT](https://github.com/Axi404/PMT)
- [SDCL](https://github.com/pascalcpp/SDCL)
- [3D CBCT Tooth Dataset]()
- [CTooth dataset]()
