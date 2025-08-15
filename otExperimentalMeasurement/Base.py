import openturns as ot
import numpy as np
import warnings
import copy
import time


class GeneralizedLeastSquareMethod:
    """
    Base class for generalized least square solvers

    Parameters
    ----------

    design : :class:`~openturns.Matrix`
        design matrix
    rhs : :class:`~openturns.Point`
        right hand side of the generalized least square problem
    rhsCovarianceMatrix : :class:`~openturns.CovarianceMatrix`
        covariance Matrix associated to the rhs
    """

    def __init__(self, design, rhs, rhsCovarianceMatrix):
        x = 1

        self.method = "SVD"

        self.rhs = rhs

        self.VY = rhsCovarianceMatrix
        self.VY_inv = (
            self.VY.inverse()
        )  # Compute inverse the covariance matrix, should be efficient as a covariance is a define positive matrix, n=inverse can be computed with Cholesky Alogirithm ?

        self.design = design
        self.GramGLS = design.transpose() * self.VY_inv * design
        self.size = self.GramGLS.getNbColumns()

    def solve(self, lambdaReg=0):
        """
        solve the generalized least square problem

        Parameters
        ----------
        lambdaReg : float - optional
            Value to be added to the Gram matrix diagonal to reguealize the matrix if needed - default 0

        Returns
        -------
        a : :class:`~openturns.Point`
            The solution

        v : :class:`~openturns.CovarianceMatrix`
            Covariance matrix of the solution
        """

        if self.method == "SVD":
            a, v = self._computeWithSVD(lambdaReg=lambdaReg)

        return a, v

    def _computeWithSVD(self, lambdaReg=0):
        """
        Internal method to solve GLS problem with SVD.

        Parameters
        ----------
        lambdaReg : float - optional
            Value to be added to the Gram matrix diagonal to reguealize the matrix if needed - default 0

        Returns
        -------
        a : :class:`~openturns.Point`
            The solution

        v : :class:`~openturns.CovarianceMatrix`
            Covariance matrix of the solution
        """

        # Compute SVD :
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
    Create a polynomial chaos by least squares with GLS or GLS-GGMR methods

    Parameters
    ----------
    inputSample : :class:`~openturns.Sample`
        The input sample.
    outputSample : :class:`~openturns.Sample`
        The output sample with dimension 1
    distribution : :class:`openturns.Distribution`
        The distribution of the input.
    VYCollection : sequence of :class:`openturns.CovarianceMatrix`
        Collection of VarianceCovariance Matrix associated to each column of outputSample
    multivariateBasis : :class:`~openturns.OrthogonalBasis`
        The orthogonal basis of functions.
    totalDegree : int
        Set the total degree of the PCE
    maximumBasisSize : int
        The maximum number of coefficients in the basis. if maximumBasisSize < basisSize(totalDegree), the n most significant coefficient are selected
    selectedIndices : :class:`~openturns.IndicesCollection` - optional
        indices of the basis functions used for the PCE - default None
    leastSquaresMethod : str
        The resolution method : "GLS" or "GGMR"

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
        Create the the functional chaos metamodel
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
            Sequence of covariance Matrix associated to each coefficient set (one per column of output Sample)
        """

        return self.result, self.covMCollection


class BatchMeanBatchCorrelation:
    """
    Base class for batch mean batch correlation

    Parameters
    ----------
    X : :class:`~openturns.Sample`
        Sample of position associated to each measurement. It can be a 1d sample if it is a time series or a 2d if it is a spatial series. larger dimension ar also handled.
    Y : :class:`~openturns.Sample`
        Sample of Measurment - Dimension must be 1
    Metrics: str, optional
        metrics used to compute distance between points. Can be "L1" or "L2". default : "L2"
    startBatchSize : int, optional
        intial batch size. Default : 2
    threshold : float, optional
        S_1 / S_0 - default 0.3
    upperThreshold : float, optional
        S_1 / S_0 - default 0.6
    fixedBatchSize : bool, optional
        if true, optimal batch size to have the desired uncorrelation is not computed. default : False
    sortSample : bool, optional
        if true, provided X and Y sample are sorted according to provided norm. First point is arbritary, then next is the closest one. Defaul : true
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
        Compute distance between point i and j according to specified norm

        Parameters
        ----------
        i : int
            point i index
        j : int
            point j index

        Returns
        -------
        d : float
            distance between i and j
        """

        if self.metrics == "L2":
            XPoint = self.X[i] - self.X[j]
            d = XPoint.norm()
            return d

    def run(self):
        """
        create the samples of batches
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
        accessor to BMBC result

        Returns
        -------
        result : :class:`BMBCResult`
        """
        return self.result

    def computeS0S1(self, M):
        """
        Compute S0S1 for a given batch size

        M : int
            Size of batch

        Returns
        -------
        S0, S1: float
            Independant and correlated variance of mean estimator
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
    Store result of BMBC algorithm

    X : :class:`~openturns.Sample`
        Sample of position associated to each measurement. It can be a 1d sample if it is a time series or a 2d if it is a spatial series. larger dimension ar also handled.
    Y : :class:`~openturns.Sample`
        Sample of Measurment - Dimension must be 1
    resultSample: :class:`~openturns.Sample`
        result sample with following columns : ["iteration","M","S0","S1, "S1/S0"]
    """

    def __init__(self, X, Y, resultSample):
        self.X = X
        self.Y = Y
        self.resultSample = resultSample

    def computeMeanEstimatorVariance(self):
        """
        compute variance of mean estimator

        Returns
        -------
        sigma2_mu : float
        """
        S0 = self.resultSample[-1, 2]
        S1 = self.resultSample[-1, 3]
        M = self.resultSample[-1, 1]
        K = self.Y.getSize() // M
        sigma2_mu = 1 / ((K - 1) * (K - 2)) * (S0 + 2 * S1)

        return sigma2_mu

    def getBatchIteration(self):
        """
        Accessor to result sample

        Returns
        -------
        SiSample : :class:`~openturns.Sample`
            Sample with values of S0, S1 for various M batch size
        """
        return self.resultSample
