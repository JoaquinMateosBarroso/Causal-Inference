import warnings
import numpy as np
from scipy.stats.contingency import crosstab


from tigramite.independence_tests.independence_tests_base import CondIndTest

from .fci import test as fci_test

class FCIndependence(CondIndTest):
    """
    Fast Conditional Independence test based on the FCI algorithm.
    
    Allows for multi-dimensional X, Y.

    Notes
    -----
    CMI and its estimator are given by

    .. math:: I(X;Y|Z) &= \sum p(z)  \sum \sum  p(x,y|z) \log
                \frac{ p(x,y |z)}{p(x|z)\cdot p(y |z)} \,dx dy dz

    Parameters
    ----------
    n_symbs : int, optional (default: None)
        Number of symbols in input data. Should be at least as large as the
        maximum array entry + 1. If None, n_symbs is inferred by scipy's crosstab.

    significance : str, optional (default: 'analytic')
        Type of significance test to use. For CMIsymb only 'fixed_thres' and
        'analytic' are available.

    sig_blocklength : int, optional (default: 1)
        Block length for block-shuffle significance test.

    conf_blocklength : int, optional (default: 1)
        Block length for block-bootstrap.

    **kwargs :
        Arguments passed on to parent class CondIndTest.
    """
    @property
    def measure(self):
        """
        Concrete property to return the measure of the independence test
        """
        return self._measure

    def __init__(self,
                 n_symbs=None,
                 significance='analytic',
                 sig_blocklength=1,
                 conf_blocklength=1,
                 **kwargs):
        # Setup the member variables
        self._measure = 'cmi_symb'
        self.two_sided = False
        self.residual_based = False
        self.recycle_residuals = False
        self.n_symbs = n_symbs
        # Call the parent constructor
        CondIndTest.__init__(self,
                             significance=significance,
                             sig_blocklength=sig_blocklength,
                             conf_blocklength=conf_blocklength,
                             **kwargs)

        if self.verbosity > 0:
            print("n_symbs = %s" % self.n_symbs)
            print("")

        if self.conf_blocklength is None or self.sig_blocklength is None:
            warnings.warn("Automatic block-length estimations from decay of "
                          "autocorrelation may not be correct for discrete "
                          "data")

    def get_analytic_significance(self, value, T, dim):
        """
        Returns p-value for shuffle significance test.

        Performes a local permutation test: x_i values are only permuted with
        those x_j for which z_i = z_j. Samples are drawn without replacement
        as much as possible.

        Parameters
        ----------
        array : array-like
            data array with X, Y, Z in rows and observations in columns.

        xyz : array of ints
            XYZ identifier array of shape (dim,).

        value : number
            Value of test statistic for original (unshuffled) estimate.

        Returns
        -------
        pval : float
            p-value.
        """
        
        
        raise NotImplementedError("Analytic significance not"+\
                                  " implemented for %s" % self.measure)
        
        