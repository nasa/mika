 # smart-nlp
***
NLP toolkit for improving coverage of design-stage risk identification.

# Installation

Please use the requirements file to install all package dependencies:

pip install --user -r requirements.txt

# Project Structure

smart_nlp (main project directory)
	|
	---readme.txt - this file
	---requirements.txt - python package requirements
	---mika/ - main MIKA toolkit code
	---test/ - test files, using python unittest
	---docs/ - documentation files
	---examples/ - examples of implementing MIKA capabilities
	---models/ - any custom BERT models used in MIKA


# Quick Start

There is a separate run script for each database available for analysis. Please run whichever script is desired.

# To Do

- trigram function shuffles word order
- can the remove pct of words test file be decommissioned?
- add tests