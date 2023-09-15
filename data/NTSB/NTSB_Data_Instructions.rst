NTSB Data Instructions
========================
Data Courtesy of: National Transportation Safety Board
 
The National Transportation Safety Board (NTSB) hosts a repository of accident reports from aviation.

The full dataset, as well as other collections, can be downloaded as a microsoft acess database via: https://data.ntsb.gov/avdata 

A subset of NTSB reports used in MIKA model training and examples is available in csv format at: https://huggingface.co/datasets/NASA-AIML/NTSB_Accidents . Note that the dataset provided is a modified format from the original public data. The data is provided as is with no warrenty or guarantees.

To download the full dataset:

1. Download each .zip file (including the "Avall.zip")
2. Move the .mdb file to a folder storing all .mdb files

Note that this process will take about an hour. 

For our examples, we were interested in only some of the database tables, which we query using SQL with pypyodbc.

To produce the dataset used in the examples, we provide a script at "ntsb_processing.py". The script produces the "ntsb_full.csv" which is used in the example.

We also use NTSB narrative data to train a custom BERT model. For this dataset, "use the ntsb_narratives_full.csv" produced from the script.