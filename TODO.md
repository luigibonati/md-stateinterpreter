# TODO list

## Ristrutturazione delle fondamenta

* [ ] MD
  1. [ ] Calcolo descrittori ( traj ) --> ( df[features], dict[residue,atom_ids, group] )
  2. [ ] Identify_states (colvar) --> df[basins,selection]
  3. [ ] Assert lenghts 
  4. [ ] Sample (1,2, chiamando 3) --> subsample

## Code

* [ ] [LUIGI] plot_relevant_features not working
* [ ] [LUIGI] plot_residues given score
* [ ] [PIETRO] classifier.plot group mode (accuracy and number of unique feats)
* [ ] [PIETRO] warm start for classifier
* [ ] [PIETRO] define groups based on:
* [ ] classifier.prune overwrite=False by default
  1. [ ] angles (ok)
  2. [ ] distances N*(N-1)/2 groups  
* [ ] sparse group lasso?
* [ ] expose classifier for out-of-sample prediction ( PLUMED interface? )
* [ ] [LUIGI] export relevant features in VMD-like style
* [ ] [LUIGI?] rewrite plot functions based on residue num and atom indices
* [ ] [PIETRO] add quadratic features to data / df ?  

## Tutorials


## Paper
* [ ] SCRIVEREEEEEEEEEEEEEEEEEEEEEEEEEE!


# Deprecated


## Model

MD.py
* [X] allow computing only subset of descriptors (e.g. via keywords)
* [X] restructure code to mimic the different steps, e.g. divide functions in:
  - states identification
  - compute descriptors
* [x] allow loading COLVAR from pd.Dataframe rather than from files
* [X] function to load features from file rather than computing them
* [X] sort minima based on CV values
* [X] compute descriptors within load traj (add keyword for list subset)
* [X] Add multiple walkers option to load multiple colvar and traj files

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
* [ ] The objects defined in classifier.py are a bit convoluted. Both of them do tiny jobs. Simplify and merge them
* [ ] Add documentation to functions

## Applications

Chignolin
* [x] analysis for tica 2d 
* [x] dihedral angles || hbonds || dihedral + hbonds
* [ ] feature importance --> residue numbers
* [ ] new analysis for biased simulation

BPTI 
* [X] analysis with diehdral angles (narjes)
* [ ] perform deep-TICA and interpret results (compare with DESHAW 2010 paper)

## Questions

* [ ] group lasso? 

