import openturns as ot
import numpy as np
import pytest
from otExperimentalMeasurement.Base import BatchMeanBatchCorrelation, BMBCResult


def test_BMBC_initialization():
    """Test initialization of BMBC class."""
    # Create sample data
    X = ot.Sample([[1.0], [2.0], [3.0], [4.0], [5.0]])
    Y = ot.Sample([[1.0], [2.0], [3.0], [4.0], [5.0]])
    
    # Initialize BMBC
    bmbc = BatchMeanBatchCorrelation(X, Y)
    
    # Check attributes
    assert bmbc.X.getSize() == 5
    assert bmbc.Y.getSize() == 5
    assert bmbc.metrics == "L2"
    assert bmbc.startBatchSize == 2
    assert bmbc.threshold == 0.3
    assert bmbc.upperThreshold == 0.6
    assert bmbc.fixedBatchSize == False
    assert bmbc.sortSample == True


def test_BMBC_with_invalid_Y_dimension():
    """Test initialization with invalid Y dimension."""
    X = ot.Sample([[1.0], [2.0], [3.0]])
    Y = ot.Sample([[1.0, 2.0], [3.0, 4.0], [5.0, 6.0]])  # Dimension 2
    
    with pytest.raises(Exception) as excinfo:
        BatchMeanBatchCorrelation(X, Y)
    assert "Dimension of Y sample must be 1" in str(excinfo.value)


def test_BMBC_computeDistance():
    """Test computeDistance method."""
    X = ot.Sample([[1.0, 2.0], [4.0, 6.0], [7.0, 8.0]])
    Y = ot.Sample([[1.0], [2.0], [3.0]])
    
    bmbc = BatchMeanBatchCorrelation(X, Y, sortSample=False)
    
    # Test L2 distance
    distance = bmbc.computeDistance(0, 1)
    expected_distance = np.sqrt((4.0 - 1.0)**2 + (6.0 - 2.0)**2)
    assert np.isclose(distance, expected_distance)


def test_BMBC_run():
    """Test run method."""
    # Create sample data
    X = ot.Sample([[i] for i in range(100)])
    Y = ot.Sample([[np.sin(i) + np.random.normal(0, 0.1)] for i in range(100)])
    
    # Initialize and run BMBC
    bmbc = BatchMeanBatchCorrelation(X, Y, fixedBatchSize=True)
    bmbc.run()
    
    # Check result
    result = bmbc.getResult()
    assert isinstance(result, BMBCResult)
    assert result.X.getSize() == 100
    assert result.Y.getSize() == 100
    assert result.resultSample.getSize() == 1


def test_BMBCResult_computeMeanEstimatorVariance():
    """Test computeMeanEstimatorVariance method."""
    X = ot.Sample([[i] for i in range(100)])
    Y = ot.Sample([[np.sin(i) + np.random.normal(0, 0.1)] for i in range(100)])
    
    bmbc = BatchMeanBatchCorrelation(X, Y, fixedBatchSize=True)
    bmbc.run()
    result = bmbc.getResult()
    
    variance = result.computeMeanEstimatorVariance()
    assert isinstance(variance, float)
    assert variance > 0


def test_BMBCResult_getBatchIteration():
    """Test getBatchIteration method."""
    X = ot.Sample([[i] for i in range(100)])
    Y = ot.Sample([[np.sin(i) + np.random.normal(0, 0.1)] for i in range(100)])
    
    bmbc = BatchMeanBatchCorrelation(X, Y, fixedBatchSize=True)
    bmbc.run()
    result = bmbc.getResult()
    
    batch_iteration = result.getBatchIteration()
    assert isinstance(batch_iteration, ot.Sample)
    assert batch_iteration.getSize() == 1


def test_BMBCResult_getBlockBoostrapSample():
    """Test getBlockBoostrapSample method."""
    X = ot.Sample([[i] for i in range(100)])
    Y = ot.Sample([[np.sin(i) + np.random.normal(0, 0.1)] for i in range(100)])
    
    bmbc = BatchMeanBatchCorrelation(X, Y, fixedBatchSize=True)
    bmbc.run()
    result = bmbc.getResult()
    
    bootstrap_sample = result.getBlockBoostrapSample()
    assert isinstance(bootstrap_sample, ot.Sample)
    assert bootstrap_sample.getSize() == 100
    assert bootstrap_sample.getDimension() == 1


def test_BMBC_with_different_metrics():
    """Test BMBC with different metrics."""
    X = ot.Sample([[1.0, 2.0], [4.0, 6.0], [7.0, 8.0]])
    Y = ot.Sample([[1.0], [2.0], [3.0]])
    
    # Test with L2 metric (only L2 is currently supported)
    bmbc_l2 = BatchMeanBatchCorrelation(X, Y, metrics="L2", sortSample=False)
    distance_l2 = bmbc_l2.computeDistance(0, 1)
    expected_distance_l2 = np.sqrt((4.0 - 1.0)**2 + (6.0 - 2.0)**2)
    assert np.isclose(distance_l2, expected_distance_l2)


def test_BMBC_with_different_parameters():
    """Test BMBC with different parameters."""
    X = ot.Sample([[i] for i in range(100)])
    Y = ot.Sample([[np.sin(i) + np.random.normal(0, 0.1)] for i in range(100)])
    
    # Test with different parameters
    bmbc = BatchMeanBatchCorrelation(
        X, Y,
        metrics="L1",
        startBatchSize=3,
        threshold=0.2,
        upperThreshold=0.5,
        fixedBatchSize=True,
        sortSample=False
    )
    
    assert bmbc.metrics == "L1"
    assert bmbc.startBatchSize == 3
    assert bmbc.threshold == 0.2
    assert bmbc.upperThreshold == 0.5
    assert bmbc.fixedBatchSize == True
    assert bmbc.sortSample == False
