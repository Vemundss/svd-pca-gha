import numpy as np 
import tqdm


def sangers_pca(x,nc=None):
    if len(x.shape) > 2:
        x = np.reshape(x, (x.shape[0],np.size(x)-x.shape[0]))
    if nc is None:
        nc = x.shape[-1]

    kernel=np.random.normal(loc=0.0, scale=1.0, size=(nc,nc))

    # center data, i.e. subtract mean
    x -= np.mean(x,axis=0)

    return _sangers_training_loop(x,kernel)


def _sangers_training_loop(x,kernel):
    mb_size = 128
    epochs = 1000
    n = 10000 # initial inverse learning rate (must be high, otherwise nan)
    n_jump = 10

    print("Sangers initial kernel: \n", np.around(kernel, decimals=2))

    for i in tqdm.trange(epochs):
        for j in range(int(x.shape[0] / 8)):
            mb = x[j*mb_size:(j+1)*mb_size]
            # forward pass
            y = np.dot(x,kernel) # dim[mb_size,E] x [E,E] => [mb_size,E]

            mb_update = _sangers_rule(x,y,kernel)
            kernel += np.mean(mb_update,axis=0)/n

            n += n_jump

    return kernel


def _sangers_rule(x: np.ndarray, y: np.ndarray, kernel: np.ndarray):
    """
    OBS! If kernel comes from tensorflow dense layer, convention is to
    use the transposed kernel.
    """
    # outer products (retain batch dim)
    y_yT = np.einsum("...i,...j->...ij", y, y)
    y_xT = np.einsum("...i,...j->...ij", y, x)

    # lower-triangular
    LT_Y = np.tril(y_yT)

    # dot product
    LT_YC = np.einsum("...ik,...kj->...ij", LT_Y, kernel)
    # LT_YC = K.dot(LT_Y, kernel)

    # mini-batch updates
    mb_update = y_xT - LT_YC

    # aggregate mini-batch update wrt. mean
    # mb_update = np.mean(mb_update, axis=0)

    return mb_update





def svd(x,nc=None):
    if len(x.shape) > 2:
        x = np.reshape(x, (x.shape[0],np.size(x)-x.shape[0]))
    if nc is None:
        nc = x.shape[-1]

    # center data, i.e. subtract mean
    x -= np.mean(x,axis=0)

    # x is a data matrix. first dim is examples, second is explanatory vars
    # i.e. dim[D,E]
    # dot(x.T,x) => [E,D] x [D, E] => [E,E]
    cov_x = np.dot(x.T,x) / x.shape[0]

    return np.linalg.svd(cov_x)


def pca():
    return False



















