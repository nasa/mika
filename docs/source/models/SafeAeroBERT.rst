SafeAeroBERT
============= 

SafeAeroBERT: A Safety-Informed Aviation-Specific Langauge Model 

Available at: https://huggingface.co/NASA-AIML/MIKA_SafeAeroBERT

base-bert-uncased model first further pre-trained on the set of Aviation Safety Reporting System (ASRS) documents up to November of 2022 and National Trasportation Safety Board (NTSB) accident reports up to November 2022. A total of 2,283,435 narrative sections are split 90/10 for training and validation, with 1,052,207,104 tokens from over 350,000 NTSB and ASRS documents used for pre-training.

The model was trained on two epochs using `AutoModelForMaskedLM.from_pretrained` with a `learning_rate=1e-5`, and total batch size of 128 for just over 32100 training steps.

An earlier version of the model was evaluted on a downstream binary document classification task by fine-tuning the model with `AutoModelForSequenceClassification.from_pretrained`. SafeAeroBERT was compared to SciBERT and base-BERT on this task, with the following performance:

.. list-table:: Classification Metrics
    :widths: 20 20 20 20 20
    :header-rows: 1
   
    * - Contributing Factor 
      - Metric
      - BERT 
      - SciBERT 
      - SafeAeroBERT
    * - Aircraft
      - Accuracy
      - **0.747**
      - 0.726
      - 0.740
    * - 
      - Precision
      - **0.716**
      - 0.691 - 
      - 0.548
    * - 
      - Recall
      - **0.747**
      - 0.726
      - 0.740
    * -
      - F-1
      - **0.719**
      - 0.699
      - 0.629
    * - Human Factors
      - Accuracy
      - **0.608**
      - 0.557
      - 0.549
    * -
      - Precision
      - **0.618**
      - 0.586
      - 0.527
    * -
      - Recall
      - **0.608**
      - 0.557
      - 0.549
    * -
      - F-1
      - **0.572***
      - 0.426
      - 0.400
    * - Procedure
      - Accuracy
      - 0.766
      - 0.755
      - **0.845**
    * -
      - Precision
      - **0.766**
      - 0.762
      - 0.742
    * -
      - Recall
      - 0.766
      - 0.755
      - **0.845**
    * -
      - F-1
      - 0.766
      - 0.758
      - **0.784**
    * - Weather
      - Accuracy
      - 0.807
      - 0.808
      - **0.871**
    * -
      - Precision
      - **0.803**
      - 0.769
      - 0.759
    * - 
      - Recall
      - 0.807
      - 0.808
      - **0.871**
    * - 
      - F-1
      - 0.805
      - 0.788
      - **0.811**


More infomation on training data, evaluation, and intended use can be found in the original publication

Citation: Sequoia R. Andrade and Hannah S. Walsh. "SafeAeroBERT: Towards a Safety-Informed Aerospace-Specific Language Model," AIAA 2023-3437. AIAA AVIATION 2023 Forum. June 2023.

