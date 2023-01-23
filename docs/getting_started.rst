Getting Started
===============

This page details how to get started with ``stateinterpreter``. 

Installation
------------
To install ``stateinterpreter``, you will need an environment with the following packages:

* ``Python3``
* ``NumPy``
* ``Pandas``
* ``Scipy``
* ``Scikit-learn``
* ``Cython``
* ``group-lasso``
* ``tdqm``
* ``Matplotlib`` (plotting functions)
* ``MDTraj`` (read MD trajectories)
* ``NGLView`` (visualize trajectories)

Once you have installed the requirements, you can install ``stateinterpreter`` by cloning the repository:
::

    git clone https://github.com/luigibonati/md-stateinterpreter.git stateinterpreter

and then installing it:

::

    cd stateinterpreter/
    pip install .

To install it in development (editable) mode:

::

    pip install -e .