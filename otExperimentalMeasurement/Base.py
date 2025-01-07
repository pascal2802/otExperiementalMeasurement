import openturns as ot


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
            a, v=self._computeWithSVD(lambdaReg=lambdaReg)

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
        Sigma_inv = ot.Matrix(self.size,self.size)
        for i in range(self.size):
            Sigma_inv[i, i] = 1.0 / SigmaDiag[i]

        GramGLS_inv = U * Sigma_inv * Vt

        a = GramGLS_inv * self.design.transpose() * self.VY_inv * self.rhs

        a = ot.Point([a[i,0] for i in range(self.size)])

        v = GramGLS_inv

        self.v = v

        self.a = a

        return a, v


class PCEWithGLS:
    """
    Create a PCE with GLS and provided Basis

    Parameters
    ----------
    design : :class:`~openturns.Matrix`
        a priori known design
    """
    def __init__(self):
        x=1
