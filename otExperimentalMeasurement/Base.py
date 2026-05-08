import openturns as ot
import numpy as np
import warnings
import copy
import time


class GeneralizedLeastSquareMethod:
    """
    Solve a generalized least squares (GLS) problem.

    The GLS method accounts for the covariance structure of the right-hand side
    to provide unbiased estimates with minimal variance.

    Parameters
    ----------
    design : :class:`~openturns.Matrix`
        Design matrix of shape (n_samples, n_features).
    rhs : :class:`~openturns.Point`
        Right-hand side vector (observations) of length n_samples.
    rhsCovarianceMatrix : :class:`~openturns.CovarianceMatrix`
        Covariance matrix of the observations, shape (n_samples, n_samples).

    Attributes
    ----------
    method : str
        Resolution method ("SVD" by default).
    GramGLS : :class:`~openturns.Matrix`
        Weighted Gram matrix: ``design.T @ VY_inv @ design``.
    a : :class:`~openturns.Point`
        Solution coefficients (available after calling :meth:`solve`).
    v : :class:`~openturns.CovarianceMatrix`
        Covariance matrix of the solution (available after calling :meth:`solve`).

    Notes
    -----
    The covariance matrix ``VY`` is inverted using Cholesky decomposition for efficiency,
    as it is positive-definite by construction.
    """

    def __init__(self, design, rhs, rhsCovarianceMatrix):
        x = 1

        self.method = "SVD"

        self.rhs = rhs

        self.VY = rhsCovarianceMatrix
        self.VY_inv = (
            self.VY.inverse()
        )  # Compute inverse the covariance matrix, should be efficient as a covariance is a define positive matrix, n=inverse can be computed with Cholesky Algorithm ?

        self.design = design
        self.GramGLS = design.transpose() * self.VY_inv * design
        self.size = self.GramGLS.getNbColumns()

    def solve(self, lambdaReg=0):
        """
        Solve the generalized least squares problem.

        The problem is formulated as:
        ``min ||design @ a - rhs||_VY_inv``,
        where ``VY_inv`` is the inverse of the observation covariance matrix.

        Parameters
        ----------
        lambdaReg : float, optional
            Tikhonov regularization parameter added to the diagonal of ``GramGLS``
            to improve numerical stability. Default is 0 (no regularization).

        Returns
        -------
        a : :class:`~openturns.Point`
            Estimated coefficients (shape: n_features).
        v : :class:`~openturns.CovarianceMatrix`
            Covariance matrix of the coefficients (shape: n_features, n_features).
            If the matrix is not symmetric, a warning is issued.
        """

        if self.method == "SVD":
            a, v = self._computeWithSVD(lambdaReg=lambdaReg)

        return a, v

    def _computeWithSVD(self, lambdaReg=0):
        """
        Internal method to solve GLS problem with SVD.

        Parameters
        ----------
        lambdaReg : float, optional
            Regularization parameter added to the diagonal of the Gram matrix
            to improve numerical stability. Default is 0.

        Returns
        -------
        a : :class:`~openturns.Point`
            The solution coefficients.
        v : :class:`~openturns.CovarianceMatrix`
            Covariance matrix of the solution.
        """

        # Compute SVD:
        GramGLS = self.GramGLS + ot.IdentityMatrix(self.size) * lambdaReg
        SigmaDiag, U, Vt = GramGLS.computeSVD(True)
        Sigma_inv = ot.Matrix(self.size, self.size)
        for i in range(self.size):
            Sigma_inv[i, i] = 1.0 / SigmaDiag[i]

        GramGLS_inv = U * Sigma_inv * Vt

        a = GramGLS_inv * self.design.transpose() * self.VY_inv * self.rhs

        a = ot.Point([a[i, 0] for i in range(self.size)])

        v = GramGLS_inv

        try:
            np.testing.assert_array_almost_equal(v, v.transpose())
            v = ot.CovarianceMatrix(v)
        except:
            warnings.warn("Covariance Matrix of estimated coefficient is not symmetric")

        self.v = v

        self.a = a

        return a, v


class PCEWithGLSorGGMR:
    """
    Build a Polynomial Chaos Expansion (PCE) metamodel using Generalized Least Squares (GLS)
    or GLS with Generalized Golgi-Morse Regularization (GGMR).

    Parameters
    ----------
    inputSample : :class:`~openturns.Sample`
        Input sample of shape (n_samples, n_input_vars).
    outputSample : :class:`~openturns.Sample`
        Output sample of shape (n_samples, 1).
    distribution : :class:`~openturns.Distribution`
        Probability distribution of the input variables.
    VYCollection : sequence of :class:`~openturns.CovarianceMatrix`
        Covariance matrices for each output column (length = outputSample.getDimension()).
    multivariateBasis : :class:`~openturns.OrthogonalBasis`
        Orthogonal basis for the PCE (e.g., Legendre, Hermite).
    totalDegree : int
        Maximum total degree of the polynomial basis.
    maximumBasisSize : int
        Maximum number of basis functions to retain. If the total basis size
        exceeds this, the least significant coefficients are discarded.
    selectedIndices : :class:`~openturns.IndicesCollection`, optional
        Pre-selected indices of basis functions to use. If None, all indices
        up to ``totalDegree`` are considered. Default is None.
    leastSquaresMethod : {"GLS", "GGMR"}, optional
        Method to solve the least squares problem. Default is "GLS".

    Attributes
    ----------
    result : :class:`~openturns.FunctionalChaosResult`
        The PCE metamodel (available after calling :meth:`run`).
    covMCollection : list of :class:`~openturns.CovarianceMatrix`
        Covariance matrices for the PCE coefficients.

    Notes
    -----
    The method automatically selects the most significant basis functions
    if ``maximumBasisSize`` is smaller than the total basis size.
    For multi-output problems (outputSample dimension > 1), only the first column
    is currently supported (see FIXME in code).
    """

    def __init__(
        self,
        inputSample,
        outputSample,
        distribution,
        VYCollection,
        multivariateBasis,
        totalDegree,
        maximumBasisSize,
        selectedIndices=None,
        leastSquaresMethod="GLS",
    ):
        self.inputSample = inputSample
        self.outputSample = outputSample
        self.distribution = distribution
        self.VYCollection = VYCollection
        self.multivariateBasis = multivariateBasis
        self.totalDegree = totalDegree
        self.maximumBasisSize = maximumBasisSize
        self.leastSquaresMethod = leastSquaresMethod
        self.selectedIndices = selectedIndices
        self.Transformation = ot.DistributionTransformation(
            self.distribution, self.multivariateBasis.getMeasure()
        )

    def run(self):
        """
        Build the PCE metamodel and compute its coefficients.

        Steps:
        1. Construct the design matrix from the input sample and basis functions.
        2. Solve the GLS problem to estimate coefficients.
        3. If necessary, truncate the basis to ``maximumBasisSize`` and re-fit.
        4. Store the result in :attr:`result` and :attr:`covMCollection`.

        Raises
        ------
        NotImplementedError
            If ``outputSample`` has dimension > 1 (not yet supported).
        """

        enumerateFunction = self.multivariateBasis.getEnumerateFunction()
        # Build an initial PC basis from the total degrees specified
        if self.selectedIndices == None:
            phisIndexes = list(
                range(enumerateFunction.getMaximumDegreeCardinal(self.totalDegree))
            )
        else:
            phisIndexes = self.selectedIndices

        phis = [
            self.multivariateBasis.build(i)
            for i in range(enumerateFunction.getMaximumDegreeCardinal(self.totalDegree))
        ]
        # Build the associated design matrix
        proxy = ot.DesignProxy(self.Transformation(self.inputSample), phis)
        design = proxy.computeDesign(
            phisIndexes
        )  # all provided basis functions are used to construct the design matrix

        # Compute a_i for a full PCE
        if self.leastSquaresMethod == "GLS":
            # FIXME : generalisation with an outputSample with a dimension larger than 1 to be done
            GLS = GeneralizedLeastSquareMethod(
                design, self.outputSample[:, 0], self.VYCollection[0]
            )
            a, v = GLS.solve(lambdaReg=1e-5)

        if a.getSize() > self.maximumBasisSize:
            phisIndexes = ot.Indices(self.maximumBasisSize)
            coeffSample = ot.Sample.BuildFromPoint(
                a[1:]
            )  # keep the a_0 coefficient in all cases
            for i in range(coeffSample.getSize()):
                coeffSample[i, 0] = abs(
                    coeffSample[i, 0]
                )  # take absolute value before sorting the coefficient
            coeffRank = (
                coeffSample.rank()
            )  # compute the rank associated to each coefficient
            phisIndexes[0] = 0  # Index associated to the mean (coefficient a_0)
            CoeffIndex = ot.Sample.BuildFromPoint(
                ot.Point(range(coeffSample.getSize()))
            )
            coeffRank.stack(CoeffIndex)
            coeffRank = coeffRank.sort()
            for i in range(
                1, self.maximumBasisSize
            ):  # only keep the coefficient with the highest value (part of variance explained)
                phisIndexes[i] = int(coeffRank[-i, 1]) + 1

            # compute a design matrix for the phis retained
            design = proxy.computeDesign(phisIndexes)

            # update coefficient values
            if self.leastSquaresMethod == "GLS":
                # FIXME : generalisation with an outputSample with a dimension larger than 1 to be done
                GLS = GeneralizedLeastSquareMethod(
                    design, self.outputSample[:, 0], self.VYCollection[0]
                )
                a, v = GLS.solve(lambdaReg=1e-5)

        # update functions collection for PCE result
        phis = [self.multivariateBasis.build(i) for i in phisIndexes]

        residualsPoint = [1.0]
        relativeErrorsPoint = [1.0]
        self.result = ot.FunctionalChaosResult(
            self.inputSample,
            self.outputSample,
            self.distribution,
            self.Transformation,
            self.Transformation.inverse(),
            self.multivariateBasis,
            phisIndexes,
            ot.Sample.BuildFromPoint(a),
            phis,
            residualsPoint,
            relativeErrorsPoint,
        )

        # FIXME : class inherited from FunctionalChaosResult to be created
        self.covMCollection = [v]

    def getResult(self):
        """
        Return the functional chaos result.

        Returns
        -------
        result : :class:`~openturns.FunctionalChaosResult`
            The metamodel.
        covMCollection : sequence of :class:`~openturns.CovarianceMatrix`
            Sequence of covariance matrices associated to each coefficient set
            (one per column of output sample).
        """

        return self.result, self.covMCollection


class BatchMeanBatchCorrelation:
    """
    Estimate the variance of the mean for correlated data using the Batch Mean method.

    This method partitions the sample into batches and adjusts the batch size
    to minimize correlation between batches, enabling consistent variance estimation.

    Parameters
    ----------
    X : :class:`~openturns.Sample`
        Sample of positions (e.g., time or spatial coordinates) of shape (n_samples, n_dims).
    Y : :class:`~openturns.Sample`
        Measurement sample of shape (n_samples, 1). Must be 1-dimensional.
    metrics : {"L1", "L2"}, optional
        Distance metric for sorting the sample. Default is "L2".
    startBatchSize : int, optional
        Initial batch size. Default is 2.
    threshold : float, optional
        Lower threshold for the ratio ``S1/S0`` (correlated vs. uncorrelated variance).
        If ``S1/S0 < threshold``, the batch size is reduced. Default is 0.3.
    upperThreshold : float, optional
        Upper threshold for ``S1/S0``. If ``S1/S0 > upperThreshold``, the batch size is increased.
        Default is 0.6.
    fixedBatchSize : bool, optional
        If True, the batch size is fixed to ``startBatchSize`` and not optimized.
        Default is False.
    sortSample : bool, optional
        If True, the sample is sorted by distance before batching. Default is True.

    Attributes
    ----------
    result : :class:`BMBCResult`
        Result object containing batch statistics (available after calling :meth:`run`).
    sampleXsorted : :class:`~openturns.Sample`
        Sorted positions (if ``sortSample=True``).
    sampleYsorted : :class:`~openturns.Sample`
        Sorted measurements (if ``sortSample=True``).

    Notes
    -----
    The method is based on the batch means approach for correlated data.
    The optimal batch size ``M`` is determined iteratively to satisfy:
    ``threshold <= S1/S0 <= upperThreshold``,
    where ``S0`` is the uncorrelated variance and ``S1`` is the correlated variance.
    """

    def __init__(
        self,
        X,
        Y,
        metrics="L2",
        startBatchSize=2,
        threshold=0.3,
        upperThreshold=0.6,
        fixedBatchSize=False,
        sortSample=True,
    ):
        self.X = copy.deepcopy(X)
        self.Y = copy.deepcopy(Y)
        if self.Y.getDimension() != 1:
            raise Exception(
                "Dimension of Y sample must be 1, provided is : %d"
                % self.Y.getDimension()
            )
        self.metrics = metrics
        self.startBatchSize = startBatchSize
        self.fixedBatchSize = fixedBatchSize
        self.sortSample = sortSample
        self.threshold = threshold
        self.upperThreshold = upperThreshold

        if self.sortSample:
            self._sortSample()
        else:
            self.sampleXsorted = self.X
            self.sampleYsorted = self.Y

    def _sortSample(self):
        sortedRank = ot.Point(self.X.getSize())
        # Take the first point of provided sample as start
        sortedRank[0] = 0
        for i in range(1, self.X.getSize()):
            X_id = sortedRank[i - 1]
            t0 = time.time()
            distanceToX_id = [
                self.computeDistance(X_id, j) for j in range(self.X.getSize())
            ]
            STemp = ot.Sample.BuildFromPoint(list(range(self.X.getSize())))
            STempX = ot.Sample.BuildFromPoint(distanceToX_id)
            STempX.stack(STemp)
            t1 = time.time()
            STempX = STempX.sort()
            t2 = time.time()

            # Remove point it self
            for k in range(self.X.getSize()):
                if STempX[k, 1] not in sortedRank:
                    sortedRank[i] = STempX[k, 1]
                    break

            t3 = time.time()

            # d_ij_min = 1e20
            # d_index = i + 1
            # for j in range(1, self.X.getSize()):
            #    # Only compute distance if point not already selected
            #    if j in sortedRank:
            #        continue
            #    d_ij = self.computeDistance(X_id, j)
            #    if d_ij < d_ij_min:
            #        d_ij_min = d_ij
            #        d_index = j
            # sortedRank[i] = d_index

        # sampleSortedRank = ot.Sample.BuildFromPoint(sortedRank)
        # sampleSortedRank.stack(self.X)
        # sampleSortedRank.stack(self.Y)
        # sampleSortedRank = sampleSortedRank.sort()

        self.sampleXsorted = ot.Sample(self.X.getSize(), self.X.getDimension())
        self.sampleYsorted = ot.Sample(self.Y.getSize(), self.Y.getDimension())
        for i in range(self.X.getSize()):
            p_id = sortedRank[i]
            self.sampleXsorted[i] = self.X[p_id]
            self.sampleYsorted[i] = self.Y[p_id]

        x = 1

    def computeDistance(self, i, j):
        """
        Compute distance between point i and j according to specified norm.

        Parameters
        ----------
        i : int
            Index of point i.
        j : int
            Index of point j.

        Returns
        -------
        d : float
            Distance between points i and j.
        """

        if self.metrics == "L2":
            XPoint = self.X[i] - self.X[j]
            d = XPoint.norm()
            return d

    def run(self):
        """
        Compute the optimal batch size and estimate the mean variance.

        Steps:

        1. If ``sortSample=True``, sort the sample by distance.
        2. Iteratively adjust the batch size ``M`` until ``S1/S0`` falls within
           [``threshold``, ``upperThreshold``].
        3. Store results in :attr:`result` (including ``S0``, ``S1``, and batch sizes).

        Returns
        -------
        None
            Results are stored in :attr:`result`.
        """

        M = self.startBatchSize

        S0, S1 = self.computeS0S1(M)

        SiSample = ot.Sample(0, 5)
        SiSample.setDescription(["iteration", "M", "S0", "S1", r"$S_1/S_0$"])
        SiSample.add([0, M, S0, S1, S1 / S0])
        n_iter = 1

        if self.fixedBatchSize == False:
            
            while (
                S1 / S0 < self.threshold or S1 / S0 > self.upperThreshold
            ) and M < self.Y.getSize() // 4:
                n_iter += 1
                if S1 / S0 < self.threshold:
                    M = int(0.7 * M)
                if S1 / S0 > self.upperThreshold:
                    M = int(1.3 * M)
                S0, S1 = self.computeS0S1(M)
                SiSample.add([n_iter, M, S0, S1, S1 / S0])

        self.result = BMBCResult(self.sampleXsorted, self.sampleYsorted, SiSample)

    def getResult(self):
        """
        Accessor to BMBC result.

        Returns
        -------
        result : :class:`BMBCResult`
            Result object containing batch statistics.
        """
        return self.result

    def computeS0S1(self, M):
        """
        Compute S0 and S1 for a given batch size.

        Parameters
        ----------
        M : int
            Size of batch.

        Returns
        -------
        S0 : float
            Independent (uncorrelated) variance of mean estimator.
        S1 : float
            Correlated variance of mean estimator.
        """

        # Create a sample with batch of K elements
        K = self.sampleYsorted.getSize() // M
        BY = ot.Sample(K, self.Y.getDimension())

        for b_id in range(K):
            batch = self.sampleYsorted[list(range(b_id * M, (b_id + 1) * M)), :]
            BY[b_id] = batch.computeMean()

        xBY = BY - self.sampleYsorted.computeMean()

        S0 = xBY.computeRawMoment(2)[0] * xBY.getSize()
        S1 = [xBY[i, 0] * xBY[i + 1, 0] for i in range(K - 1)]
        S1 = sum(S1)

        return S0, S1


class BMBCResult:
    """
    Store result of the Batch Mean Batch Correlation (BMBC) algorithm.

    Parameters
    ----------
    X : :class:`~openturns.Sample`
        Original positions of shape (n_samples, n_dims).
    Y : :class:`~openturns.Sample`
        Original measurements of shape (n_samples, 1).
    resultSample : :class:`~openturns.Sample`
        Sample containing iteration history with columns:
        ["iteration", "M", "S0", "S1", "S1/S0"].

    Attributes
    ----------
    X : :class:`~openturns.Sample`
        Original positions.
    Y : :class:`~openturns.Sample`
        Original measurements.
    resultSample : :class:`~openturns.Sample`
        Iteration history (last row = final result).
    """

    def __init__(self, X, Y, resultSample):
        self.X = X
        self.Y = Y
        self.resultSample = resultSample

    def computeMeanEstimatorVariance(self):
        """
        Compute variance of mean estimator.

        Returns
        -------
        sigma2_mu : float
            Estimated variance of the mean.
        """
        S0 = self.resultSample[-1, 2]
        S1 = self.resultSample[-1, 3]
        M = self.resultSample[-1, 1]
        K = self.Y.getSize() // M
        sigma2_mu = 1 / ((K - 1) * (K - 2)) * (S0 + 2 * S1)

        return sigma2_mu

    def getBatchIteration(self):
        """
        Accessor to result sample.

        Returns
        -------
        resultSample : :class:`~openturns.Sample`
            Sample with values of S0, S1 for various batch sizes.
        """
        return self.resultSample

    def getBlockBoostrapSample(self):
        """
        Get a block bootstrap sample from the initial provided sample.

        Returns
        -------
        Yb : :class:`~openturns.Sample`
            Bootstrap sample constructed by resampling blocks of size M.
        """
        M = int(self.resultSample[-1, 1])
        K = int(self.Y.getSize() // M)
        # build a sample of block indices
        blockPoint = ot.Point(list(range(K)))
        blockIndices = ot.Sample().BuildFromPoint(blockPoint)
        bootStrapExp = ot.BootstrapExperiment(blockIndices)
        generatedBlockIndices = bootStrapExp.generate()
        Yb = ot.Sample(K * M, self.Y.getDimension())
        for i in range(K):
            b_id = int(generatedBlockIndices[i, 0])
            Yb[list(range(i * M, (i + 1) * M))] = self.Y[
                list(range(b_id * M, (b_id + 1) * M))
            ]
            # for j in range(i * M, (i + 1) * M):
            #    indexInBlock = j - i * M
            #    Yb[j] = self.Y[b_id + indexInBlock]

        return Yb
