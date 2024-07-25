Custom Named-Entity Recognition
================================

Available at: https://huggingface.co/NASA-AIML/MIKA_BERT_FMEA_NER

base-bert-uncased model first further pre-trained then fine-tuned for custom NER to extract failure-relevant entities from incident and accident reports. 
The model was trained on manually annotated NASA LLIS reports and evaluated on SAFECOM reports. 

NER model training was for 4 epochs with:`BertForTokenClassification.from_pretrained` , `learning_rate=2e-5`, ` weight_decay=0.01,` 

The model was trained to identify the following long-tailed entities:

- CAU: failure cause
- MOD: failure mode
- EFF: failure effect
- CON: control process
- REC: recommendations

Performace:

.. list-table:: Classification Metrics
    :widths: 20 20 20 20 20
    :header-rows: 1

    * - Entity 
      - Precision
      - Recall 
      - F-1 
      - Support
    * - CAU 
      - 0.31
      - 0.19
      - 0.23
      - 1634 
    * - CON
      - 0.49
      - 0.34
      - 0.40
      - 3859
    * - EFF
      - 0.45
      - 0.20
      - 0.28
      - 1959 
    * - MOD
      - 0.19
      - 0.52
      - 0.28
      - 594 
    * - REC
      - 0.30
      - 0.59
      - 0.40
      - 954
    * - Average
      - 0.41
      - 0.32
      - 0.33
      - 9000

More infomation on training data, evaluation, and intended use can be found in the original publication


Citation:
S. R. Andrade and H. S. Walsh, "What Went Wrong: A Survey of Wildfire UAS Mishaps through Named Entity Recognition," 2022 IEEE/AIAA 41st Digital Avionics Systems Conference (DASC), Portsmouth, VA, USA, 2022, pp. 1-10, doi: 10.1109/DASC55683.2022.9925798.
https://ieeexplore.ieee.org/abstract/document/9925798

