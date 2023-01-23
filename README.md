stateinterpreter
==============================
[//]: # (Badges)
[![GitHub Actions Build Status](https://github.com/luigibonati/md-stateinterpreter/workflows/CI/badge.svg)](https://github.com/luigibonati/md-stateinterpreter/actions?query=workflow%3ACI)
[![codecov](https://codecov.o/gh/luigibonati/md-stateinterpreter/branch/main/graph/badge.svg)](https://codecov.io/gh/luigibonati/md-stateinterpreter/branch/main)


Supporting code and material for the publication:

_Novelli, Bonati, Pontil and Parrinello_, Characterizing Metastable States with the Help of Machine Learning, [JCTC](https://pubs.acs.org/doi/10.1021/acs.jctc.2c00393) (2022) 

## Getting started

To install ``stateintepreter`` first clone the repository:
```
git clone https://github.com/luigibonati/md-stateinterpreter.git
```

Install the dependencies, e.g. by:

```
cd stateinterpreter
pip install -r requirements.txt
```

Install the ``stateintepreter`` package:
```
pip install .
```

### Additional requirement: mlcvs

In order to identify the slow collective variables we use the TICA-based CVs , which we optimize using the [mlcvs](https://github.com/luigibonati/mlcvs) package. Please see the [instructions](https://mlcvs.readthedocs.io/en/latest/getting_started.html) on how to install it.

## Tutorials

Check out the [tutorials](https://github.com/luigibonati/md-stateinterpreter/tree/main/tutorials) to see how to use the functionalities of ``stateinterpreter``. 
 
Project based on the 
[Computational Molecular Science Python Cookiecutter](https://github.com/molssi/cookiecutter-cms) version 1.6.
