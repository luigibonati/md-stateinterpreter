# TODO list

## Model

MD.py (restructure code)
* [X] allow computing only subset of descriptors (e.g. via keywords)
* [X] restructure code to mimic the different steps, e.g. divide functions in:
  - states identification
  - compute descriptors
* [x] allow loading COLVAR from pd.Dataframe rather than from files
* [X] function to load features from file rather than computing them
* [X] sort minima based on CV values
* [X] compute descriptors within load traj (add keyword for list subset)

FES
* [x] extend compute FES function to allow weighted data
* [x] less memory intensive FES function? (especially high dimensions)
  1. [x] implement KDE
  2. [x] how to find bandwith? (careful for weighted data and rules)
  3. [x] optimization scheme to find local minima

Plot
* [x] move plotting functions to module
* [] default name of the states displayed in the title should be consistent with the basin index used in classification (useful for states subsets)

classifier.py
* [x] add function to save results to file (list of relevant features)
* [x] add keyword to sample to select subset of basins
* [] The objects defined in classifier.py are a bit convoluted. Both of them do tiny jobs. Simplify and merge them

Questions
* [X] question: features for quadratic CVs should be obtained from combination of features within each state or all of them? e.g. state1: x1,x2 - state2: x3,x4 --> x1 x2, x3 x4 OR x1 x2, x1 x3, x1 x4, x2 x3 ecc... ? --> ALL OF THEM
* [X] question: how to check that model is robust? (e.g. repeat N times? ) --> CONVEX
* [X] question: sometimes (especially with chignolin, hbonds) throws the warning: `ConvergenceWarning: The max_iter was reached which means the coef_ did not converge` should we try to increase max_iter? --> pass `max_iter` in dict to compute

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


