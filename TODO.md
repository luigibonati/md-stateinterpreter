# TODO list

## Model

MD.p
* [ ] allow computing only subset of descriptors (e.g. via keywords)
* [ ] restructure code to mimic the different steps, e.g. divide functions in:
  - states identification
  - compute descriptors
* [ ] allow loading COLVAR from pd.Dataframe rather than from files
* [ ] extend compute FES function to allow weighted data
* [ ] less memory intensive FES function? (especially high dimensions)


classifier.py
* [x] add function to save results to file (list of relevant features)
* [ ] add plot of number of features per state and total number of unique features as function of C
* [ ] question: features for quadratic CVs should be obtained from combination of features within each state or all of them? e.g. state1: x1,x2 - state2: x3,x4 --> x1 x2, x3 x4 OR x1 x2, x1 x3, x1 x4, x2 x3 ecc... ?
* [ ] question: how to check that model is robust? (e.g. repeat N times? )
* [ ] question: sometimes (especially with chignolin, hbonds) throws the warning: `ConvergenceWarning: The max_iter was reached which means the coef_ did not converge` should we try to increase max_iter?

## Systems

Chignolin
* [x] analysis for tica 2d 
* [x] analysis for tica 3d
* [x] dihedral angles || hbonds || dihedral + hbonds
* [ ] feature importance --> residue numbers
* [ ] new analysis for biased simulation

BPTI 
* [ ] analysis with diehdral angles (narjes)
* [ ] perform deep-TICA and interpret results (compare with DESHAW 2010 paper)


