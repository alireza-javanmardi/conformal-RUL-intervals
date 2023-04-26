# Conformal Prediction Intervals for Remaining Useful Lifetime Estimation

[![arXiv](https://img.shields.io/badge/arXiv-2302.12238-b31b1b.svg)](https://arxiv.org/abs/2212.14612)

This repository contains the code for the paper: [Conformal Prediction Intervals for Remaining Useful Lifetime
Estimation](https://arxiv.org/pdf/2212.14612.pdf),
written by Alireza Javanmardi and Eyke Hüllermeier.
This paper is going to appear in the International Journal of Prognostics and Health Management (IJPHM) 2023.

As the paper's title suggests, we are interested in estimating the remaining useful lifetime (RUL) of a system. However, rather than providing a specific point in time for system failure, such as 

**"The system will fail in $5$ cycles (or days, weeks, etc.)"**,

the output will present a range of potential failure times, such as 

**"The system will fail between $3$ and $6$ cycles (or days, weeks, etc.)"**.

Here is the general procedure of how to construct conformal prediction intervals using any arbitrary single-point RUL estimator:

![image](conformal-prediction.png "general procedure of CP for RUL estimation")
## Setup
1. Clone the repository
2. Create a new virtual environment and install the requirements:
```shell
 pip install -r requirements.txt
```
3. Activate the virtual environment and run:
  ```shell
 python CNN_experiment.py CMAPSS1 0.1 22
 ```
This line of code will perform an experiment on the CMAPSS dataset FD001 using a deep convolutional neural network as the single-point RUL estimator. 10% of the training data will be put aside for calibration, and the random seed will be set to 22. 

## Citation

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
