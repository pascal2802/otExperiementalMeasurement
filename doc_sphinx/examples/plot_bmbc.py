"""
Batch Mean Batch Correlation on a correlated time series 
++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
"""

# # Batch Mean Batch Correlation example
# Method to compute uncertainty of empiric mean estimator when sample is not i.i.d.


# Import librairies
import openturns as ot
import otExperimentalMeasurement as otEM
import matplotlib.pyplot as plt

# Generate correlated time series
N = 1000000
# Assuming one measure each second
X = ot.Sample.BuildFromPoint(list(range(N)))
Y = ot.Sample(N, 1)
autoCorrelationCoeff = 0.9
distR = ot.Uniform(0, 1)
RSample = distR.getSample(N)
Y[0, 0] = 1
for i in range(1, N):
    Y[i, 0] = (
        Y[i - 1, 0] * autoCorrelationCoeff
        + (1.0 - autoCorrelationCoeff) * RSample[i, 0]
    )

# mixingDistribution = ot.KPermutationsDistribution(N, N)
# newIndices = mixingDistribution.getRealization()
# X = ot.Sample([X[i] for i in newIndices])

# ## visualization of time series
fig = plt.figure()
ax = fig.add_subplot(1, 1, 1)
ax.plot(X[:, 0], Y[:, 0])
fig.show()

# ## Batch Mean Batch Correlation
M = 20
threshold = 0.2
bmbc = otEM.BatchMeanBatchCorrelation(
    X, Y, threshold=threshold, fixedBatchSize=False, startBatchSize=M, sortSample=False
)
bmbc.run()
result = bmbc.getResult()
var_mu = result.computeMeanEstimatorVariance()
print(r"$N \sigma^2_{mu}$ : %.4f - expected : 0.0834" % (N * var_mu))

# Plot S1/S0 evolution
resultSample = result.getBatchIteration()
fig = plt.figure(num="SiRatio")
ax = fig.add_subplot(1, 1, 1)
ax.plot(resultSample[:, 0], resultSample[:, 4])
ax.set_xlabel(resultSample.getDescription()[0])
ax.set_ylabel(resultSample.getDescription()[4])
fig.show()

# ## BlockBoostrap 
# Generate sample using bootstrap method
bSample = result.getBlockBoostrapSample()
print(bSample)
