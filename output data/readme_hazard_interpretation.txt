Results interpretation steps

First open all necessary documents:

1. Open the pyLDAvis output for the given results 
(ex: ICS_full_combined_topics-May-17-2021/combined text_hldavis.html)
2. Open the hlda results output
(ex: ICS_full_combined_topics-May-17-2021/hlda_results.xlsx)
3. Open the raw text of the dataset
(ex: ICS_raw_text_filtered.csv)
4. Open a spread sheet for recording results
(ex: hazard_interpretation_template)

The hazard_interpreation template has two sheets: hazard-focused and topic-focused.
The topic-focused sheet is self explainatory. 
The hazard-focused sheet once filled should have one row per hazard, which contains 
various information. This is basically pulling relevant words from topics and 
aggregating to a hazard.

For example, we could have the following results for a row:

hazard words -> tortoise, beetle, species; endangered, endangered species habitat, habitat

***NOTE: when multiple words are required to define a hazard, such as both "speces" and "endangered",
separate the two type of lists with a semicolon
ex1: ground; heli, helicopter, aircraft, airtanker, tanker, aerial
ex2: highway, road, traffic; close, closure, impacts
ex3: resource, crew; limited, share, lack, fatigue
****

hazard level 1 topics -> 10, 6
hazard level 2 topics -> 33, 26, 17, 11
Best documents (for each of the topics) -> 2006_1224_MULTIPLE JUNE FIRES,
2006_AK-DAS-612166_JARVIS CREEK, 2006_00276_MILLER COMPLEX, 2014_VAVAS1406043_PINE CREEK
2014_WA-WFS-513_SAND RIDGE, 2014_96_BEAR
hazard category -> environmental
hazard name -> ecosystem considerations

See "hazard_interpretation_test" for an example of a filled out hazard sheet

Next, begin interpretation:
0. From the pyLDAvis output, go through each topic, for each topic:
1. Use the hlda_results to find the best document for the topic
2. Find the raw text of the best document (sheet in step 3 above)
3. Identify if there are any relevant hazard words/categories from examining
both the best document and the top 30 words in the visualization
4. Fill out all relevant information in the spread sheet for recording results. 
This includes both topic-focused and hazard-focused information.

Once the spread sheet has been filled and all topics have been examined, the hazard list
can be complied from the hazard-focused results in the spread sheet. At this point,
the hazard category and hazard name sections can be filled in.

Finally, we need to be able to identify these hazards in the documents for trend analysis.
Since the topics are mixed-quality, may contain extraneous words, and may have multiple topics 
with the same hazards, we cannot use just the raw topics as the equivalent of a hazard. 
Instead, we will use the "Hazard words" in the hazard-focused results spread sheet to identify
occurences for each hazard found.*

See "trend_analysis_functions.py" for the functions of how this is implemented.

*note: this method is not perfect and is prone to errors, future work may develop better methods
