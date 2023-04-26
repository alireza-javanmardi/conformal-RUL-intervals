# Conformal Prediction Intervals for Remaining Useful Lifetime Estimation

[![arXiv](https://img.shields.io/badge/arXiv-2302.12238-b31b1b.svg)](https://arxiv.org/abs/2212.14612)

This repository contains the code for the paper: [Conformal Prediction Intervals for Remaining Useful Lifetime
Estimation](https://arxiv.org/pdf/2212.14612.pdf),
written by Alireza Javanmardi and Eyke Hüllermeier.
This paper is going to appear in the International Journal of Prognostics and Health Management (IJPHM) 2023.

As the paper's title suggests, we are interested in estimating remaining useful lifetime (RUL) of a system. However, instead of providing single-point estimations, we provide intervals as output. In other words, instead of saying 

**The system will fail in $5$ cycles (or days, weeks, etc.)**, 

we say

**The system will fail between $3$ and $6$ cycles (or days, weeks, etc.)**.

Here is the general prodecure of how to construct conformal prediction intervals using any arbitrary single-point RUL estimator: 

![image](conformal-prediction.png "general procedure of CP for RUL estimation")

## Installation
1. Clone the repository
2. Create a new virtual environment and isntall the requirements:
```shell
    pip install -r requirements.txt
```
3. Activate the virtual environment and run:
  ```shell
 python CNN_experiment.py CMAPSS1 0.1 10
 ```
This will run the code for CMAPSS dataset FD001 using CNN as an underlying regression model with a calibration portion of 0.1 and a random seed of 10.

## Citing

If you use this code, please cite our paper:

```
@misc{javanmardi2022conformal,
      title={Conformal Prediction Intervals for Remaining Useful Lifetime Estimation}, 
      author={Alireza Javanmardi and Eyke Hüllermeier},
      year={2022},
      eprint={2212.14612},
      archivePrefix={arXiv},
      primaryClass={cs.LG}
}

```
