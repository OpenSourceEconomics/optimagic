"""Functions for inferences in maximum likelihood models."""
import numpy as np

from estimagic.exceptions import INVALID_INFERENCE_MSG
from estimagic.utilities import robust_inverse


def cov_hessian(hess):
    """Covariance based on the negative inverse of the hessian of loglike.

    While this method makes slightly weaker statistical assumptions than a covariance
    estimate based on the outer product of gradients, it is numerically much more
    problematic for the following reasons:

    - It is much more difficult to estimate a hessian numerically or with automatic
      differentiation than it is to estimate the gradient / jacobian
    - The resulting hessian might not be positive definite and thus not invertible.

    Args:
        hess (numpy.ndarray): 2d array hessian matrix of dimension (nparams, nparams)

    Returns:
       numpy.ndarray: covariance matrix (nparams, nparams)


    Resources: Marno Verbeek - A guide to modern econometrics :cite:`Verbeek2008`

    """
    info_matrix = -1 * hess
    cov_hes = robust_inverse(info_matrix, msg=INVALID_INFERENCE_MSG)

    return cov_hes


def cov_jacobian(jac):
    """Covariance based on outer product of jacobian of loglikeobs.

    Args:
        jac (numpy.ndarray): 2d array jacobian matrix of dimension (nobs, nparams)

    Returns:
        numpy.ndarray: covariance matrix of size (nparams, nparams)


    Resources: Marno Verbeek - A guide to modern econometrics.

    """
    info_matrix = np.dot((jac.T), jac)
    cov_jac = robust_inverse(info_matrix, msg=INVALID_INFERENCE_MSG)

    return cov_jac


def cov_robust(jac, hess):
    """Covariance of parameters based on HJJH dot product.

    H stands for Hessian of the log likelihood function and J for Jacobian,
    of the log likelihood per individual.

    Args:
        jac (numpy.ndarray): 2d array jacobian matrix of dimension (nobs, nparams)
        hess (numpy.ndarray): 2d array hessian matrix of dimension (nparams, nparams)


    Returns:
        numpy.ndarray: covariance HJJH matrix (nparams, nparams)

    Resources:
        https://tinyurl.com/yym5d4cw

    """
    info_matrix = np.dot((jac.T), jac)
    cov_hes = cov_hessian(hess)
    sandwich_cov = np.dot(cov_hes, np.dot(info_matrix, cov_hes))

    return sandwich_cov


def se_from_cov(cov):
    """Standard deviation of parameter estimates based on the function of choice.

    Args:
        cov (numpy.ndarray): Covariance matrix

    Returns:
        standard_errors (numpy.ndarray): 1d array with standard errors

    """
    standard_errors = np.sqrt(np.diag(cov))

    return standard_errors


def cov_cluster_robust(jac, hess, design_options):
    """Cluster robust standard errors.

    A cluster is a group of observations that correlate amongst each other,
    but not between groups. Each cluster is seen as independent. As the number
    of clusters increase, the standard errors approach robust standard errors.

    Args:
        jac (np.array): "jacobian" - an n x k + 1-dimensional array of first
            derivatives of the pseudo-log-likelihood function w.r.t. the parameters
        hess (np.array): "hessian" - a k + 1 x k + 1-dimensional array of
            second derivatives of the pseudo-log-likelihood function w.r.t.
            the parameters
    Returns:
        cluster_robust_se (np.array): a 1d array of k + 1 standard errors
        cluster_robust_var (np.array): 2d variance-covariance matrix

    """
    cluster_meat = _clustering(design_options, jac)
    cluster_robust_var = _sandwich_step(hess, cluster_meat)
    return cluster_robust_var


def cov_strata_robust(jac, hess, design_options):
    """Cluster robust standard errors.

    A stratum is a group of observations that share common information. Each
    stratum can be constructed based on age, gender, education, region, etc.
    The function runs the same formulation for cluster_robust_se for each
    stratum and returns the sum. Each stratum contain primary sampling units
    (psu) or clusters. If observations are independent, but wish to have to
    strata, make the psu column take the values of the index.

    Args:
        jac (np.array): "jacobian" - an n x k + 1-dimensional array of first
            derivatives of the pseudo-log-likelihood function w.r.t. the parameters
        hess (np.array): "hessian" - a k + 1 x k + 1-dimensional array of
            second derivatives of the pseudo-log-likelihood function w.r.t.
            the parameters
        design_options (pd.DataFrame): dataframe containing psu, stratum,
            population/design weight and/or a finite population corrector (fpc)

    Returns:
        strata_robust_se (np.array): a 1d array of k + 1 standard errors
        strata_robust_var (np.array): 2d variance-covariance matrix

    """
    strata_meat = _stratification(design_options, jac)
    strata_robust_var = _sandwich_step(hess, strata_meat)
    return strata_robust_var


def _sandwich_step(hess, meat):
    """The sandwich estimator for variance estimation.

    This is used in several robust covariance formulae.

    Args:
        hess (np.array): "hessian" - a k + 1 x k + 1-dimensional array of
            second derivatives of the pseudo-log-likelihood function w.r.t.
            the parameters
        meat (np.array): the variance of the total scores

    Returns:
        se (np.array): a 1d array of k + 1 standard errors
        var (np.array): 2d variance-covariance matrix

    """
    invhessian = np.linalg.inv(hess)
    var = np.dot(np.dot(invhessian, meat), invhessian)
    return var


def _clustering(design_options, jac):
    """Variance estimation for each cluster.

    The function takes the sum of the jacobian observations for each cluster.
    The result is the meat of the sandwich estimator.

    Args:
        design_options (pd.DataFrame): dataframe containing psu, stratum,
            population/design weight and/or a finite population corrector (fpc)
        jac (np.array): "jacobian" - an n x k + 1-dimensional array of first
            derivatives of the pseudo-log-likelihood function w.r.t. the parameters

    Returns:
        cluster_meat (np.array): 2d square array of length k + 1. Variance of
            the likelihood equation (Pg.557, 14-10, Greene 7th edition)

    """

    list_of_clusters = design_options["psu"].unique()
    meat = np.zeros([len(jac[0, :]), len(jac[0, :])])
    for psu in list_of_clusters:
        psu_scores = jac[design_options["psu"] == psu]
        psu_scores_sum = psu_scores.sum(axis=0)
        meat += np.dot(psu_scores_sum[:, None], psu_scores_sum[:, None].T)
    cluster_meat = len(list_of_clusters) / (len(list_of_clusters) - 1) * meat
    return cluster_meat


def _stratification(design_options, jac):
    """Variance estimatio for each strata stratum.

    The function takes the sum of the jacobian observations for each cluster
    within strata. The result is the meat of the sandwich estimator.

    Args:
        design_options (pd.DataFrame): dataframe containing psu, stratum,
            population/design weight and/or a finite population corrector (fpc)
        jac (np.array): "jacobian" - an n x k + 1-dimensional array of first
            derivatives of the pseudo-log-likelihood function w.r.t. the parameters

    Returns:
        strata_meat (np.array): 2d square array of length k + 1. Variance of
        the likelihood equation

    """
    n_params = len(jac[0, :])
    stratum_col = design_options["strata"]
    # Stratification does not require clusters
    if "psu" not in design_options:
        design_options["psu"] = design_options.index
    else:
        pass
    psu_col = design_options["psu"]
    strata_meat = np.zeros([n_params, n_params])
    # Variance estimation per stratum
    for stratum in stratum_col.unique():
        psu_in_strata = psu_col[stratum_col == stratum].unique()
        psu_jac = np.zeros([n_params])
        if "fpc" in design_options:
            fpc = design_options["fpc"][stratum_col == stratum].unique()
        else:
            fpc = 1
        # psu_jac stacks the sum of the observations for each cluster.
        for psu in psu_in_strata:
            psu_jac = np.vstack([psu_jac, np.sum(jac[psu_col == psu], axis=0)])
        psu_jac_mean = np.sum(psu_jac, axis=0) / len(psu_in_strata)
        if len(psu_in_strata) > 1:
            mid_step = np.dot(
                (psu_jac[1:] - psu_jac_mean).T, (psu_jac[1:] - psu_jac_mean)
            )
            strata_meat += (
                fpc * (len(psu_in_strata) / (len(psu_in_strata) - 1)) * mid_step
            )
        # Apply "grand-mean" method for single unit stratum
        elif len(psu_in_strata) == 1:
            strata_meat += fpc * np.dot(psu_jac[1:].T, psu_jac[1:])

    return strata_meat