Documentation
=================

Here goes a high-level description of the code structure.

Core modules
------------

.. rubric:: Descriptors (``stateinterpreter.descriptors``)

.. currentmodule:: stateinterpreter.descriptors

.. autosummary::
   :toctree: autosummary
   :template: custom-class-template.rst

   compute_descriptors
   load_descriptors

.. rubric:: Metastable (``stateinterpreter.metastable``)

.. currentmodule:: stateinterpreter.metastable

.. autosummary::
   :toctree: autosummary
   :template: custom-class-template.rst

   identify_metastable_states
   approximate_FES

.. rubric:: ML (``stateinterpreter.ml``)

.. currentmodule:: stateinterpreter.ml

.. autosummary::
   :toctree: autosummary
   :template: custom-class-template.rst

   prepare_training_dataset
   Classifier

Utilities
---------

.. rubric:: Input/Output (``stateinterpreter.utils.io``)

.. currentmodule:: stateinterpreter.utils.io

.. autosummary::
   :toctree: autosummary
   :template: custom-class-template.rst

   load_dataframe
   load_trajectory

.. rubric:: Visualize (``stateinterpreter.utils.visualize``)

.. currentmodule:: stateinterpreter.utils.visualize

.. autosummary::
   :toctree: autosummary
   :template: custom-class-template.rst

   visualize_features
   compute_residue_score
   visualize_residue_score

.. rubric:: Plot (``stateinterpreter.utils.plot``)

.. currentmodule:: stateinterpreter.utils.plot

.. autosummary::
   :toctree: autosummary
   :template: custom-class-template.rst

   plot_states
   plot_regularization_path
   plot_classifier_complexity_vs_accuracy
   plot_combination_states_features
   plot_states_features
   plot_histogram_features


