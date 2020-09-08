import errno
import os


def mkdir_p(path):
    """
    Creates directory with parent directory as needed
    (similar to 'mkdir -p ${path}').
    Does not raise an error if directory exists

    Inputs:
    -------
    path:   type(str)
    """
    try:
        os.makedirs(path)
    except OSError as err:
        if err.errno == errno.EEXIST and os.path.isdir(path):
            pass
        else:
            raise err
    return
