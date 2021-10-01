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
	---module/ - main project code
	---test/ - test files, using python unittest

# Quick Start

There is a separate run script for each database available for analysis. Please run whichever script is desired.

# To Do

- some tests fail - lda/hlda tests have issues with the folders they reference, preprocessing fails on line 392 - appears to be an issue with empty docs after certain preprocessing steps
- __remove_words_in_pct_of_docs needs a test - possible bug
- condense test files so you only have to run things once
