Batch Mean Batch Correlation (BMBC) Theory
===========================================

This page explains the theoretical foundations of the Batch Mean Batch Correlation (BMBC) algorithm
for estimating the variance of the mean estimator in correlated time series.

Introduction
------------

The BMBC algorithm is designed to compute the uncertainty of empirical mean estimators when dealing
with correlated data, particularly in time series analysis. Traditional methods assume independent and
identically distributed (i.i.d.) samples, but real-world data often exhibits temporal correlation that
must be accounted for in uncertainty quantification.

Theoretical Background
-----------------------

For a time series :math:`Y_1, Y_2, \ldots, Y_N` with temporal correlation, the empirical mean is:

.. math::
    \bar{Y} = \frac{1}{N} \sum_{i=1}^N Y_i

The variance of this mean estimator :math:`\sigma^2_{\mu}` is not simply :math:`\sigma^2/N` as in the
i.i.d. case, but must account for the correlation structure.

Batch Means Method
------------------

The batch means method divides the time series into :math:`K` batches of size :math:`M`:

.. math::
    K = \left\lfloor \frac{N}{M} \right\rfloor

For each batch :math:`b` (where :math:`b = 1, 2, \ldots, K`), we compute the batch mean:

.. math::
    \bar{Y}_b = \frac{1}{M} \sum_{i=(b-1)M + 1}^{bM} Y_i

The variance between batch means provides an estimate of the variance of the overall mean.

Batch Correlation Analysis
---------------------------

The BMBC algorithm extends the batch means method by analyzing the correlation between consecutive
batches. We compute two key quantities:

1. **S₀ - Independent variance component**:

.. math::
    S_0 = \frac{1}{K-1} \sum_{b=1}^K (\bar{Y}_b - \bar{\bar{Y}})^2

2. **S₁ - Correlated variance component**:

.. math::
    S_1 = \frac{1}{K-1} \sum_{b=1}^{K-1} (\bar{Y}_b - \bar{\bar{Y}})(\bar{Y}_{b+1} - \bar{\bar{Y}})

where :math:`\bar{\bar{Y}}` is the mean of the batch means.

Optimal Batch Size Selection
----------------------------

The algorithm adaptively selects the batch size :math:`M` to achieve a target ratio :math:`S_1/S_0`
that indicates appropriate decorrelation between batches. The target range is typically between 0.2 and 0.6.

Variance Estimation
-------------------

The final variance estimate of the mean is:

.. math::
    \hat{\sigma}^2_{\mu} = \frac{1}{(K-1)(K-2)} (S_0 + 2S_1)

This estimator accounts for both the independent and correlated components of variance.

Block Bootstrap
---------------

For uncertainty quantification, the algorithm provides a block bootstrap method that resamples entire
batches rather than individual observations, preserving the correlation structure.

References
----------

The BMBC algorithm is based on the following reference [BMBC2023]_.

.. [BMBC2023] Russo, S., Luchini, P., 2017. A fast algorithm for the estimation of statistical error in DNS (or experimental) time averages. Journal of Computational Physics 347, 328–340. https://doi.org/10.1016/j.jcp.2017.07.005 

Implementation Details
----------------------

The OpenTURNS implementation provides:

- Adaptive batch size selection
- Automatic correlation detection
- Block bootstrap resampling
- Integration with OpenTURNS statistical framework

See Also
--------

- :class:`~otExperimentalMeasurement.BatchMeanBatchCorrelation` - Main implementation class
- :class:`~otExperimentalMeasurement.BMBCResult` - Result storage and analysis
- :doc:`/auto_examples/plot_bmbc` - Practical example with code
