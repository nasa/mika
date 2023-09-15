LLIS Data Instructions
========================
The NASA lessons learned information system is available to the public at: https://llis.nasa.gov/

NASA-internal personnel can retrieve an excel file of the data via the NASA Enginnering Network.

Non-NASA users can obtain the data via the webscraper.

Webscraper requirements are:
 - requests
 - selenium (https://selenium-python.readthedocs.io/)
 - beautiful soup
 - time
 - pandas

Notes about webscraping:
 - The LLIS pages require at least a one second time delay between scraping to allow the data to load.
 - The webscraper provided uses the chrome webdriver, which must be installed in the location where the webscraper runs.
 - The webscraper opens a chrome page for every lesson, which can be disruptive. Keeps this in mind.
 - There around about 2,100 documents in the LLIS and the document numbers do not correspond directly to the count, so we added a csv file containing all valid lesson ID numbers to speed webscraping.
 - To scrape the entire LLIS, the estimated run time is around a day. 
 - It is recommended to run the webscraper with the test lesson set prior to scraping the entire dataset to ensure it runs properly.
 - The webscraped version of the LLIS will require more data cleaning than the NASA-internal version, specifically with endline characters ("\n").