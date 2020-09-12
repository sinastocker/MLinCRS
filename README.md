# MLinCRS

Code collection to fit KRR ML models

### Requirements
Requirements for Example_tutorial.ipynb are:

* python            >= 3.7
* ase               >= 3.19.1
* numpy             >= 1.19.0
* scikit-learn      >= 0.23.1
* jupyter-notebook  >= 6.1.3
* matplotlib        >= 3.3.0

* mltools           https://github.com/simonwengert/mltools.git
* QUIP, quippy, GAP https://libatoms.github.io/GAP/installation.html

### Data formats
Molecular database stored in an extended .xyz file. This file serves as
input for the preparation of the kernel matrices.
The molecule info line has to contain an `id` and `AE` keyword, which is
the id of the molecule in the database and the atomization energy.

Reaction network with bond breaking reactions of types A - > B + C
(reaction type I) or A -> B (reaction type II) stored in a .txt file.
Each row is a sepatate reaction. The file consists of 4 columns.

|Column  | Content                                                                                      |
|--------|----------------------------------------------------------------------------------------------|
|1       | Index of molecule A (molecule id in .xyz file)                                               |
|2       | Index of molecule B (molecule id in .xyz file)                                               |
|3       | Index of molecule C (molecule id in .xyz file), reaction type I or "[x]", reaction type II   |
|4       | DFT calculated Reaction energy (RE) [eV], RE = sum(AE(products)) - AE(educt)                 |

### Authors
Sina Stocker (sina.stocker@tum.de)
