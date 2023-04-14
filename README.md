# conformal-RUL-intervals

This repository contains the code for the paper: [Conformal Prediction Intervals for Remaining Useful Lifetime
Estimation](https://arxiv.org/pdf/2212.14612.pdf),
written by Alireza Javanmardi and Eyke Hüllermeier.
This paper is going to appear in the International Journal of Prognostics and Health Management (IJPHM) 2023.

As the paper's title suggests, we are interested in estimating remaining useful lifetime (RUL) of a system. However, instead of providing single-point estimations, we provide intervals as output. In other words, instead of saying 

**The system will fail in $5$ cycles (or days, weeks, etc.)**, 

we say

**The system will fail between $3$ and $6$ cycles (or days, weeks, etc.)**.

To utilize the code for CMAPSS dataset FD001 using CNN as an underlying regression model with a calibration portion of 0.1 and a random seed of 10, one may execute the following code:

`python CNN_experiment.py CMAPSS1 0.1 10`
