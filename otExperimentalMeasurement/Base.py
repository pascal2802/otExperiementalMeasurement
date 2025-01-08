import openturns as ot
import numpy as np
import warnings


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
    multivariateBasis : :class:`~openturns.OrthogonalBasis`maxim
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
            # update functions collection for PCE result
            phis = [self.multivariateBasis.build(i) for i in phisIndexes]

            # update coefficient values
            if self.leastSquaresMethod == "GLS":
                # FIXME : generalisation with an outputSample with a dimension larger than 1 to be done
                GLS = GeneralizedLeastSquareMethod(
                    design, self.outputSample[:, 0], self.VYCollection[0]
                )
                a, v = GLS.solve(lambdaReg=1e-5)

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

    def getResult(self):
        """
        Return the functional chaos result.

        Returns
        -------
        result : ot.FunctionalChaosResult
            The metamodel.
        """

        return self.result
