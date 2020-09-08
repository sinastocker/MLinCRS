import numpy as np
from collections import deque


def read_reaction_network_g0(filename, highest_id_molecules):
    """
    Function that reads a reaction network file of the form
    A -> B + C in which C can be either a molecule or "[x]".

    Inputs:
    -------
    filename : str
        Name of the reaction network file.
    highest_id_molecules : int
        Highest id of molecule that is contained
        in the reaction network.

    Returns:
    --------
    e : np.array, N
        Array with all indices of the educt molecules A.
    p1 : np.array, N
        Array with all indices of the product molecules B.
    p2 : np.array, N
        Array with all indices of the product molecules C.
        If the reaction is of type A -> B, the index is set
        to highest_id_molecules + 1.
    Y_re : np.array, N
        Reaction energies.
    """
    # Transfrom highest_id_molecules in an integer
    highest_id_molecules = int(highest_id_molecules)
    # Read the educt indices
    e = np.loadtxt(filename, delimiter=',', usecols=[0, ]).astype(int)
    # Read the product indices from molecules B
    p1 = np.loadtxt(filename, delimiter=',', usecols=[1, ]).astype(int)
    # Read the product indices from molecules C
    p2_store = np.loadtxt(filename, delimiter=',', usecols=[2, ], dtype=str)

    # Check whether reaction is of type A -> B + C or A -> B.
    p2 = deque()
    for p_i in p2_store:
        try:
            p = int(p_i)
        except ValueError:
            # if reaction is of type A -> B, set index to
            # highest_id_molecules + 1
            p = highest_id_molecules + 1
        p2.append(p)
    p2 = np.asarray(p2)

    # Read reaction energie
    Y_re = np.loadtxt(filename, delimiter=',', usecols=[3, ])
    return e, p1, p2, Y_re


def append_energy_4_rearrangements(E):
    """
    Appends an zero energy value for
    rearrangement reactions.

    Inputs:
    -------
    E : np.array, N
        Energy array for which a zero value should be added.
    """
    return np.hstack((E, 0))
