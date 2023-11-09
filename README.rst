Overview
========

**MIKA** (Manager for Intelligent Knowledge Access) is a toolkit intended to assist design-time risk 
analysis and safety assurance via advanced natural language processing capabilities. 

The full documentation is available at: https://nasa.github.io/mika/ 

State-of-the-art natural language processing (NLP) techniques enable new ways to access safety-relevant 
knowledge available in text-based documents. MIKA packages advanced NLP techniques and uses models 
specially trained for engineering applications to allow engineers to better tap into knowledge available in
safety reports, accident reports, incident reports, lessons learned documents, and other engineering 
docuements.

To this end, the MIKA open-source toolkit has been developed for the following uses:

#. Enabling rapid exploration of a set of text-based engineering text documents

#. Analyzing large, unstructured datasets, or exploiting structure in data when it is available 
   (flexibility)

#. Increasing the value of engineering documents through adding metadata, analyses, and summaries

Key Features
------------
MIKA includes two key capabilties, Knowledge Discovery and Information Retrieval, for exploring text-based 
repositories. Both use BERT models as a backbone for multiple functions. 

Knowledge Discovery (KD) enables the user to extract useful, meaningful information from narrative-based 
engineering documents. This includes both supervised and unsupervised methods, such as:

   #. A variety of topic modeling methods

   #. Custom named-entity recognition extraction of a Failure Modes and Effect Analysis (FMEA)-style table

   #. The ability to analyze trends in hazards or failures

Information Retrieval (IR) enables the user to search a set of documents and obtain relevant documents 
or passages according to their query. This includes:

   #. An information retrieval pipeline using a bi-encoder and cross-encoder with options for users to 
      choose from pretrained or custom models

Installation
---------------

MIKA is available on PyPI and can be installed with:

.. code-block:: python
    pip install nasa-mika

After installing mika, initialize nltk by running the following in python:

.. code-block:: python
    import nltk
    nltk.download('words')

Now you can import anything in MIKA:
.. code-block:: python
    from mika.kd import FMEA
    from mika.kd import Topic_Model_plus
    from mika.kd.trend_analysis import *
    from mika.kd.NER import *
    from mika.ir import search

    from mika.utils import Data
    from mika.utils.SAFECOM import *
    from mika.utils.SAFENET import *
    from mika.utils.LLIS import *
    from mika.utils.ICS import *

The latest version of MIKA is also available via the NASA github page using:

.. code-block:: python
    git clone https://github.com/nasa/mika.git

Prerequisites
-------------
MIKA uses Python 3 and has been tested on python>=3.8. We recommend installing pytorch via anaconda first and configuring it for GPU use if desired. If installing via pip, all prerequesits are included.

Alternatively, you can manually clone MIKA and install the requirements. MIKA requires the following packages and their dependencies outlined in requirements.txt:

.. code-block:: python

    BERTopic
    datasets
    gensim
    matplotlib
    nltk
    numpy
    octis
    pandas
    pathlib
    pingouin
    pkg_resources
    pyLDAvis
    regex
    scikit-learn
    scipy
    seaborn
    sentence-transformers
    spacy
    symspellpy
    tomotopy
    torch
    transformers
    wordcloud

These can be installed with pip.

Additional packages that should be downloaded for optional functions include:

.. code-block:: python
    
    graphvis #(to plot hierarchical topic models)
    pickle   #(to save results)
    jupyter notebook #(to view examples in the repository)

Support
-------
MIKA is considered research code and is under development to refine features, add new capabilities, and 
improve workflows. Certain functions may change over time. Please contact the contributors if any bugs or 
issues are present.

Contributors
------------
`Hannah Walsh <https://github.com/walshh>`_ : Semantic Search capability, Custom Information Retrieval 
capability, Topic Model Plus, Data utility, Documentation

`Sequoia Andrade <https://github.com/sequoiarose>`_ : FMEA capability, custom NER, Trend Analysis, Topic
Model Plus, Data utilty, Dataset-specific utilities, Code Review, Documentation


Notices
-------

Copyright © 2023 United States Government as represented by the Administrator of the National Aeronautics and Space Administration.  All Rights Reserved.

Disclaimers
~~~~~~~~~~~

No Warranty: THE SUBJECT SOFTWARE IS PROVIDED "AS IS" WITHOUT ANY WARRANTY OF ANY KIND, EITHER EXPRESSED, IMPLIED, OR STATUTORY, INCLUDING, BUT NOT LIMITED TO, ANY WARRANTY THAT THE SUBJECT SOFTWARE WILL CONFORM TO SPECIFICATIONS, ANY IMPLIED WARRANTIES OF MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE, OR FREEDOM FROM INFRINGEMENT, ANY WARRANTY THAT THE SUBJECT SOFTWARE WILL BE ERROR FREE, OR ANY WARRANTY THAT DOCUMENTATION, IF PROVIDED, WILL CONFORM TO THE SUBJECT SOFTWARE. THIS AGREEMENT DOES NOT, IN ANY MANNER, CONSTITUTE AN ENDORSEMENT BY GOVERNMENT AGENCY OR ANY PRIOR RECIPIENT OF ANY RESULTS, RESULTING DESIGNS, HARDWARE, SOFTWARE PRODUCTS OR ANY OTHER APPLICATIONS RESULTING FROM USE OF THE SUBJECT SOFTWARE.  FURTHER, GOVERNMENT AGENCY DISCLAIMS ALL WARRANTIES AND LIABILITIES REGARDING THIRD-PARTY SOFTWARE, IF PRESENT IN THE ORIGINAL SOFTWARE, AND DISTRIBUTES IT "AS IS."

Waiver and Indemnity:  RECIPIENT AGREES TO WAIVE ANY AND ALL CLAIMS AGAINST THE UNITED STATES GOVERNMENT, ITS CONTRACTORS AND SUBCONTRACTORS, AS WELL AS ANY PRIOR RECIPIENT.  IF RECIPIENT'S USE OF THE SUBJECT SOFTWARE RESULTS IN ANY LIABILITIES, DEMANDS, DAMAGES, EXPENSES OR LOSSES ARISING FROM SUCH USE, INCLUDING ANY DAMAGES FROM PRODUCTS BASED ON, OR RESULTING FROM, RECIPIENT'S USE OF THE SUBJECT SOFTWARE, RECIPIENT SHALL INDEMNIFY AND HOLD HARMLESS THE UNITED STATES GOVERNMENT, ITS CONTRACTORS AND SUBCONTRACTORS, AS WELL AS ANY PRIOR RECIPIENT, TO THE EXTENT PERMITTED BY LAW.  RECIPIENT'S SOLE REMEDY FOR ANY SUCH MATTER SHALL BE THE IMMEDIATE, UNILATERAL TERMINATION OF THIS AGREEMENT. 


