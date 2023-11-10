Topic Modeling Results
----------------------

MIKA supports multiple topic modeling algorthims and outputs results in a standard format. Results are typically saved with topic numbers, topic words, documents per topic, and best document per topic. 

Just the topic model results can be saved, as well as a full set of results with a taxonomy, document-topic distribution, and coherence. To save just the topics:

``tm.save_bert_topics()``, ``tm.save_lda_topics()``, or  ``tm.save_hlda_topics()``

To save the full set of results:

``tm.save_bert_results()``, ``tm.save_lda_results()``, or  ``tm.save_hlda_results()``

.. csv-table:: LDA results
   :file: ../examples/lda_example.csv
   :header-rows: 1

.. csv-table:: hLDA results
   :file: ../examples/hlda_example.csv
   :header-rows: 1

.. csv-table:: BERTopic results
   :file: ../examples/bertopic_example.csv
   :header-rows: 1

