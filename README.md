# conformal-RUL-intervals

This repository contains the code for the paper: [Conformal Prediction Intervals for Remaining Useful Lifetime
Estimation](https://arxiv.org/pdf/2212.14612.pdf),
written by Alireza Javanmardi and Eyke Hüllermeier.
This paper is going to appear in the International Journal of Prognostics and Health Management (IJPHM) 2023.

As the paper's title suggests, we are interested in estimating remaining useful lifetime (RUL) of a system. However, instead of providing single-point estimations, we provide intervals as output. In other words, instead of saying 

**The system will fail in $5$ cycles (or days, weeks, etc.)**, 

we say

**The system will fail between $3$ and $6$ cycles (or days, weeks, etc.)**.

To be able to use the code, for instance, for dataset FD001 in CMAPSS with 0.1 being the portion of data used for calibration and random seed being set to 10, one can simply run the following code:

`python CNN_experiment.py CMAPSS1 0.1 10`
