"""
Batch Mean Batch Correlation on a correlated time series
============================================================

This example demonstrates the Batch Mean Batch Correlation (BMBC) method
to compute uncertainty of empirical mean estimator when samples are not i.i.d.
"""

# %%
# Setup and imports
# ------------------

# %%
import openturns as ot
import otExperimentalMeasurement as otEM
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

# %%
# Generate correlated time series
# --------------------------------

# %%
# Create a correlated time series with N=100000 points
N = 1000000
X = ot.Sample.BuildFromPoint(list(range(N)))
Y = ot.Sample(N, 1)

# Generate autocorrelated data with coefficient 0.9
autoCorrelationCoeff = 0.9
distR = ot.Uniform(0, 1)
RSample = distR.getSample(N)
Y[0, 0] = 1
for i in range(1, N):
    Y[i, 0] = (
        Y[i - 1, 0] * autoCorrelationCoeff
        + (1.0 - autoCorrelationCoeff) * RSample[i, 0]
    )

# %%
# Apply Batch Mean Batch Correlation
# ----------------------------------

# %%
# Initialize BMBC with adaptive batch sizing
M = 300  # Initial batch size
threshold = 0.2  # Lower threshold for S1/S0 ratio
upperThreshold = 0.6  # Upper threshold for S1/S0 ratio

bmbc = otEM.BatchMeanBatchCorrelation(
    X, Y, 
    threshold=threshold, 
    upperThreshold=upperThreshold,
    fixedBatchSize=False, 
    startBatchSize=M, 
    sortSample=False
)

# %%
# Run the algorithm
bmbc.run()
result = bmbc.getResult()

# %%
# Compute and display variance of mean estimator
var_mu = result.computeMeanEstimatorVariance()
print(r"Variance of mean estimator: $N \sigma^2_{\mu}$ = %.4f" % (N * var_mu))
print("Expected value for comparison: ~0.0834")

# %%
# Analyze iteration results
# -------------------------

# %%
# Get iteration results
resultSample = result.getBatchIteration()
print(f"\nNumber of iterations: {resultSample.getSize()}")
print("Result columns:", resultSample.getDescription())

# %%
# Display detailed iteration information
print("\nIteration details:")
for i in range(resultSample.getSize()):
    row = resultSample[i]
    print(f"Iteration {int(row[0])}: M={int(row[1])}, S0={row[2]:.3f}, S1={row[3]:.3f}, S1/S0={row[4]:.4f}")

# %%
# Visualize S1/S0 evolution
# --------------------------

# %%
# Create the plot
fig = plt.figure(figsize=(10, 6))
ax = fig.add_subplot(1, 1, 1)

# Plot S1/S0 ratio evolution
ax.plot(resultSample[:, 0], resultSample[:, 4], 'bo-', 
         linewidth=2, markersize=8, label='S1/S0 ratio')

# Add threshold lines
ax.axhline(y=threshold, color='r', linestyle='--', label='Lower threshold')
ax.axhline(y=upperThreshold, color='g', linestyle='--', label='Upper threshold')

# Formatting
ax.set_xlabel('Iteration')
ax.set_ylabel('S1/S0 ratio')
ax.set_title('Convergence of S1/S0 Ratio in BMBC Algorithm')
ax.legend()
ax.grid(True, alpha=0.3)
plt.tight_layout()

# %%
# Save the plot
output_file = '/tmp/bmbc_s1_s0_evolution.png'
plt.savefig(output_file, dpi=150, bbox_inches='tight')
print(f"\nPlot saved to {output_file}")

# %%
# Block Bootstrap demonstration
# -----------------------------

# %%
# Generate bootstrap sample using optimal batch size
bootstrapSample = result.getBlockBoostrapSample()
print(f"\nBlock bootstrap sample generated with {bootstrapSample.getSize()} points")
print("First 5 values:", [bootstrapSample[i, 0] for i in range(5)])
print("Last 5 values:", [bootstrapSample[i, 0] for i in range(-5, 0)])
