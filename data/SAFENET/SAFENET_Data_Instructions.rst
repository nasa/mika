SAFENET Data Instructions
=========================
Data Courtesy of: National Interagency Fire Center

The SAFENET reporting system is use to report hazards, mishaps, and near misses during ground crew operations in wildfire fighting.

There are a total of 2481 reports at the time of these instructions.

The dataset can be obtained as an excel form via: https://safenet.nifc.gov/safenet_rpt.cfm 

A subset of SAFENET reports used in MIKA model training and examples is available in csv format in the repository under `data/SAFENET` . Note that the dataset provided is a modified format from the original public data. The data is provided as is with no warrenty or guarantees.

The website can only export so many reports at once, so we recommend downloading each year separately.

To do this, on the website provided:

1. Select "Criteria #1" available fields to be "Year"
2. Select the desired year. We recommend downloading oldest to newest and starting with 1989.
3. Hit "Create SAFENET Report Button". This loads the query from 2.
4. Hit "Export to Excel"
5. Repeat for all available years.

After downloading all of the years of reports as excel files, place them all in one folder.

We have provided a processing script to combine all these documents into one file using the folder path, availble at "SAFENET_processing.py".