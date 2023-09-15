ASRS Data Instructions
========================

Data Courtesy of: Aviation Safety Reporting System

The Aviation Safety Reporting System is a reporting system for aviation operations and is available here: https://asrs.arc.nasa.gov/ 

There are over 100,000 ASRS reports in the system. 

A subset of ASRS reports used in MIKA model training and examples is available in csv format at: https://huggingface.co/datasets/NASA-AIML/ASRS . Note that the dataset provided is a modified format from the original public data. The data is provided as is with no warrenty or guarantees.

To obtain the full ASRS, you must download the reports from https://akama.arc.nasa.gov/ASRSDBOnline/QueryWizard_Filter.aspx 

There is a limit to how many reports can be exported at once (5000), so it is recommended to download reports one year at a time, in six month increments. For some years, the data will still exceed the limit and you will have to use smaller increments (3-4 months). This is a manual process that takes about 1-2 hours.

To download the data:

1. Select "Date of Incident". This will move the query to "current search".
2. In the "current search" select the start and end date. For example: begin date - 2022 january, end date - 2022 june
3. Hit "save".
4. Hit "Run Search".
5. Export results as a CSV.
6. Repeat until all years/months of data have been saved.

Once the data has all been downloaded and saved, add all the files to one folder. We have added a script to combine all the data into one csv availble at "ASRS_combine_data.py". The script also has functions for replacing abbreviations with full text if desired. 
