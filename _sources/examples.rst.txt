Examples
=========

Example notebooks and scripts can be found in in the base repository under `/examples`. The examples directory is separated into `/examples/kd` and `/examples/ir`, with a complete case stufy of all the capabilities of MIKA under `/examples/Case_Study_NTSB`

Utils
+++++

The main components of utils are the Data class and dataset-specific functions, which are used throughout other examples. A dedicated Data example can be found under  `/examples/utils`

.. toctree::
    nblinks/data_example.nblink

IR 
+++

Information Retrieval examples can be found in `/examples/IR`, with two sub directories for corresponding publications.

.. toctree::
    nblinks/ir_example.nblink

KD 
+++

Knowledge Discovery examples can be found in `/examples/KD`, with sub directories for Hazard Extraction and Analysis of Trends (HEAT), Failure Modes and Effects Analysis (FMEA), and failure taxonomies. Within these subdirectories, there are separate directories that correspond to specific publications.

.. toctree::
    nblinks/topic_model_plus.nblink
    results/topic_modeling_results
    results/taxonomy_results
    nblinks/fmea_example.nblink
    results/fmea_results
    nblinks/ics_heat.nblink
    nblinks/safecom_heat.nblink
    


NTSB Case Study 
++++++++++++++++

The NTSB Case Study is documented in the publication:

Andrade, S. and Walsh, H. (2023), MIKA: Manager for Intelligent Knowledge Access Toolkit for Engineering Knowledge Discovery and Information Retrieval. INCOSE International Symposium, 33: 1659-1673. https://doi.org/10.1002/iis2.13105

.. toctree::
   nblinks/ntsb.nblink