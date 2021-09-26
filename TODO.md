# TODO list

## Model

MD.py
* extend compute FES function to allow weighted data
* less memory intensive FES function? (especially high dimensions)
* allow loading COLVAR from pd.Dataframe rather than from files

classifier.py
* add plot of number of features per state and total number of unique features as function of C
* question: features for quadratic CVs should be obtained from combination of features within each state or all of them? e.g. state1: x1,x2 - state2: x3,x4 --> x1 x2, x3 x4 OR x1 x2, x1 x3, x1 x4, x2 x3 ecc... ?

## Systems

Chignolin
* analysis for tica 2d and 3d
* dihedral angles || hbonds || dihedral + hbonds
* feature importance --> residue numbers
* new analysis for biased simulation

BPTI 
* analysis with diehdral angles (narjes)
* perform deep-TICA and interpret results (compare with DESHAW 2010 paper)

