**MIKA** (Manager for Intelligent Knowledge Access) is a toolkit intended to assist design-time risk analysis and safety assurance via advanced natural language processing capabilities. Emerging natural language processing techniques enable new ways to access safety-relevant knowledge available in text-based documents; however, limited tools exist that leverage state-of-the-art natural language processing capabilities in a way that is accessible for new users, enables convenient application of multiple techniques, and is open source. Moreover, natural language processing for technical documentation often requires specialized techniques which are not always available in other toolkits. The MIKA toolkit fills this need by providing a variety of easy-to-use advanced NLP techniques centered around engineering documents.

Overview
====================================

Safety reports and other engineering documents contain a wealth of information from past projects and operations that could improve system safety and design, yet they are often under utilized. Advances in natural language processing techniques have improved information extraction and retrieval in consumer technology, biomedicine, and finance, for instance, but have not been applied to engineering documents on the same scale. To this end, the Manager for Intelligent Knowledge Access (MIKA) open-source toolkit has been developed for the following core capabilities:

1.	Enable rapid exploration of a set of text-based engineering text documents.
2.	Analyze large, unstructured datasets, or exploit structure in data when it is available (flexibility).
3.	Increase the value of engineering documents through adding metadata, analyses, and summaries. 

Key Features
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
MIKA includes two key capabilties for exploring text-based repositories which use BERT models as a backbone for multiple functions: 

- Knowledge Discovery (KD) enables the user to extract useful, meaningful information from text-based engineering documentation. This includes both supervised and unsupervised methods, such as:
    - a variety of topic modeling methods via the topic_model_plus class
    - custom named-entity recognition for FMEA extraction
    - the ability to analyze trends in hazards or failures using the trend_analysis module
- Information Retrieval (IR) enables the user to search a set of documents and obtain relevant documents or passages according to their query. This includes:
    - a default search class using a fine-tuned sentence-BERT model 
    - a custom information retrieval class that allows a user to fine-tune their own model

MIKA is considered research code and is under development to refine features, add new capabilities, and improve workflows. With this in mind, certain functions may change overtime. Please contact the contributors if any bugs or issues are present.

Getting Started
====================================
The latest version of MIKA is currently available via the NASA github and can be downloaded from the MIKA github page using:
:: 
    git clone https://github.com/nasa/mika.git

Prerequisites
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
MIKA operates on Python 3 and requires the following packages and their dependencies outlined in requirements.txt:
::
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

Additional packages that should be downloaded for optional functions include:
::
    graphvis #(to plot heigherarchical topic models)
    pickle   #(to save results)
    jupyter notebook #(to view examples in the repository)

Contributors
====================================
`Hannah Walsh <https://github.com/walshh>`_ : Semantic Search capability, Custom Information Retrieval capability, Topic Model Plus, Data utility

`Sequoia Andrade <https://github.com/sequoiarose>`_ : FMEA capability, custom NER, Trend Analysis, Topic Model Plus, Data utilty, dataset-specific utilities, code review


Notices:
====================================

Copyright © 2022 United States Government as represented by the Administrator of the National Aeronautics and Space Administration.  All Rights Reserved.

Disclaimers
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

No Warranty: THE SUBJECT SOFTWARE IS PROVIDED "AS IS" WITHOUT ANY WARRANTY OF ANY KIND, EITHER EXPRESSED, IMPLIED, OR STATUTORY, INCLUDING, BUT NOT LIMITED TO, ANY WARRANTY THAT THE SUBJECT SOFTWARE WILL CONFORM TO SPECIFICATIONS, ANY IMPLIED WARRANTIES OF MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE, OR FREEDOM FROM INFRINGEMENT, ANY WARRANTY THAT THE SUBJECT SOFTWARE WILL BE ERROR FREE, OR ANY WARRANTY THAT DOCUMENTATION, IF PROVIDED, WILL CONFORM TO THE SUBJECT SOFTWARE. THIS AGREEMENT DOES NOT, IN ANY MANNER, CONSTITUTE AN ENDORSEMENT BY GOVERNMENT AGENCY OR ANY PRIOR RECIPIENT OF ANY RESULTS, RESULTING DESIGNS, HARDWARE, SOFTWARE PRODUCTS OR ANY OTHER APPLICATIONS RESULTING FROM USE OF THE SUBJECT SOFTWARE.  FURTHER, GOVERNMENT AGENCY DISCLAIMS ALL WARRANTIES AND LIABILITIES REGARDING THIRD-PARTY SOFTWARE, IF PRESENT IN THE ORIGINAL SOFTWARE, AND DISTRIBUTES IT "AS IS."

Waiver and Indemnity:  RECIPIENT AGREES TO WAIVE ANY AND ALL CLAIMS AGAINST THE UNITED STATES GOVERNMENT, ITS CONTRACTORS AND SUBCONTRACTORS, AS WELL AS ANY PRIOR RECIPIENT.  IF RECIPIENT'S USE OF THE SUBJECT SOFTWARE RESULTS IN ANY LIABILITIES, DEMANDS, DAMAGES, EXPENSES OR LOSSES ARISING FROM SUCH USE, INCLUDING ANY DAMAGES FROM PRODUCTS BASED ON, OR RESULTING FROM, RECIPIENT'S USE OF THE SUBJECT SOFTWARE, RECIPIENT SHALL INDEMNIFY AND HOLD HARMLESS THE UNITED STATES GOVERNMENT, ITS CONTRACTORS AND SUBCONTRACTORS, AS WELL AS ANY PRIOR RECIPIENT, TO THE EXTENT PERMITTED BY LAW.  RECIPIENT'S SOLE REMEDY FOR ANY SUCH MATTER SHALL BE THE IMMEDIATE, UNILATERAL TERMINATION OF THIS AGREEMENT. 


