import numpy as np
from MLinCRS.helpers import mkdir_p
from mltools.gap import Gap


# KRR
def get_krr_coeff(K, y, default_sigma, y_sigma=None, rcond=None, mc=None):
    """
    Fit Kernel Ridge regression model.

    Parameters:
     -----------
    K : np.ndarray (N, N)
        The kernel matrix representing the similarities
        between individual elements.
    y : np.ndarray (N)
        quantity you want to fit.
    default_sigma : float
        Regularization parameter. Will be overwritten
        if y_sigma is set.
    y_sigma : np.ndarray (N), optional
        If given it represents the specified regularization
        parameter for each individual element.
    rcond : None or int
        Parameter from np.linalg.lstsq parameter
    mc : float, optional
        Mean value of the training set properties.
        default=None

    Returns:
    --------
    coeff : np.ndarray (N)
        Fitted krr coefficients.
    """
    if mc:
        y = y - mc
    if not isinstance(y_sigma, type(None)):
        default_sigma = y_sigma
    coeff = np.linalg.lstsq(K + default_sigma*np.eye(K.shape[0]), np.asarray(y), rcond=rcond)[0]
    return coeff


def get_prediction(K_test, coeff, mc=None):
    """
    Predict quantities using the kernel ridge coeffs.

    K_test : np.ndarray (N, M)
        The kernel matrix representing the similarities
        between training and test elements.
    coeff : np.ndarray (N, 1)
        Fitted krr coefficients.
    mc : float, optional
        Mean value of the training set properties.
        default=None

    Returns:
    --------
    y_predict : np.ndarray(N)
        Returns predicted values.
    """
    y_predict = np.dot(K_test, coeff)
    if mc:
        return y_predict + mc
    return y_predict


def grid_search_1d_validation_set(
        K_train,
        K_val,
        y_train,
        y_val,
        sigmas,
        destination='./Results_validation_set',
        mc=None
):
    """
    Function to create hypersurfaces for given grid parameter. The hypersurfaces are generated for the training and
    validation set. The results are written to a txt file called hypersurface_data.txt in the destination folder.

    K_train : np.ndarray (N, N)
        The kernel matrix representing the similarities
        between N training elements.
    K_val : np.ndarray (M, N)
        The kernel matrix representing the similarities
        between N training elements and M configurations to be predicted.
    y_train : np.ndarray (N, 1)
        Training properties.
    y_val : np.ndarray (N, 1)
        Validation set properties.
    sigmas: list
        Grid with regularization parameters.
    destination : str, optional
        The final output will be saved in the
        specified location.
        default=Results_validation_set
    mc : float, optional
        Mean value of the training set properties.
        default=None
    """
    # Set the Gap instance
    gap = Gap()
    # prepare kernels and properties
    mkdir_p(destination)

    # Loop over regularization parameter
    rmsd_t, rmsd_v = [], []
    for sig in sigmas:
        # train the model
        alpha = get_krr_coeff(K_train, y_train, default_sigma=sig, mc=mc)
        # Get predicted values for the training set
        pred_train = get_prediction(K_train, alpha, mc=mc)
        # Get predicted values for the vaidation set
        pred_val = get_prediction(K_val, alpha, mc=mc)
        # Calculate RMSEs
        rmsd_t.append(gap.get_rmse(y_train, pred_train))
        rmsd_v.append(gap.get_rmse(y_val, pred_val))

    # Concatenating data
    data = np.hstack((np.asarray(sigmas)[:, None], np.asarray(rmsd_t)[:, None], np.asarray(rmsd_v)[:, None]))
    # Save the hyperparameter surface for plotting
    np.savetxt('{}/hypersurface_data.txt'.format(destination), data, header='sigma, train, validation')
