SAFECOM Data Instructions
=========================
Data Courtesy of: Department of the Interior

The Aviation Safety Communiqué or SAFECOM system is available to the public at: https://www.safecom.gov/ 

A subset of SAFECOM reports used in MIKA model training and examples is available in csv format in the repository under `data/SAFECOM` . Note that the dataset provided is a modified format from the original public data. The data is provided as is with no warrenty or guarantees.

There is no easy to download file of the data, so it must be obtained via webscraping for any analysis.

Webscraper requirements are:
 - requests
 - selenium (https://selenium-python.readthedocs.io/)
 - beautiful soup
 - time
 - pandas

Notes about webscraping:
 - The SAFECOM pages require at least a one second time delay between scraping to allow the data to load.
 - The webscraper provided uses the chrome webdriver, which must be installed in the location where the webscraper runs.
 - The webscraper opens a chrome page for every document, which can be disruptive. Keeps this in mind.
 - There around about 23,000 documents in the SAFECOM and the document numbers do not correspond directly to the count, so some of the scraping will be for invalid numbers. To speed up the process, the highest number per year is included in the webscraping script.
 - To scrape the entire SAFECOM, the estimated run time is around two days. 
 - The webscraper tends to crash every so often, likely due to poor network connection or any interruptions. We recommend running the script in an IDE that saves the variables (spyder or jupyter will work, VScode may also work) so that you can save your progress data frame to a csv after a crash. Lines 94-97 can be uncommented to continue scraping at your last saved document after a crash, then you can append the saved pre-crash csv to the new csv using lines 108-109. Overall, it is recommended to scrape in increments and save progress to prevent data loss during a crash.