Custom Information Retrieval
============================

Available at: https://huggingface.co/NASA-AIML/MIKA_Custom_IR

This is a `sentence-transformers`_ model: It maps sentences & paragraphs to a 768 dimensional dense vector space and can be used for tasks like clustering or semantic search.

.. _sentence-transformers: https://www.SBERT.net

The model is custom trained on engineering documents for asymmetric infromation retrieval. It is intended to be used to identify engineering documents relevant to a query for use in design time. For example, a repository can be queried to find support for requirements or learn more about a specific type of failure.

Usage (Sentence-Transformers) 
+++++++++++++++++++++++++++++

Using this model becomes easy when you have `sentence-transformers`_ installed:

.. code-block:: python

    pip install -U sentence-transformers

Then you can use the model like this:

.. code-block:: python

    from sentence_transformers import SentenceTransformer
    sentences = ["This is an example sentence", "Each sentence is converted"]

    model = SentenceTransformer("NASA-AIML/MIKA_Custom_IR")
    embeddings = model.encode(sentences)
    print(embeddings)



Evaluation Results
+++++++++++++++++++

This model was evaluated on three queries using precision at k for k=10,20, and 30. Mean average precision (MAP) was also calculated. The model was baselines against the pre-trained SBERT.

.. list-table:: Evaluation
   :widths: 50 50
   :header-rows: 1

   * - IR Method
     -  MAP 
   * - Pre-trained sBERT
     - 0.648
   * - Fine-tuned sBERT
     - 0.807

Training
+++++++++

The model was trained with the parameters:

**DataLoader**:

`sentence_transformers.datasets.NoDuplicatesDataLoader.NoDuplicatesDataLoader` of length 693 with parameters:
.. code-block:: python

    {'batch_size': 32}


**Loss**:

`sentence_transformers.losses.MultipleNegativesRankingLoss.MultipleNegativesRankingLoss` with parameters:
 .. code-block:: python

  {'scale': 20.0, 'similarity_fct': 'cos_sim'}

Parameters of the fit()-Method:

.. code-block:: python

    {
    "epochs": 2,
    "evaluation_steps": 100,
    "evaluator": "sentence_transformers.evaluation.InformationRetrievalEvaluator.InformationRetrievalEvaluator",
    "max_grad_norm": 1,
    "optimizer_class": "<class 'transformers.optimization.AdamW'>",
    "optimizer_params": {"lr": 2e-05},
    "scheduler": "WarmupLinear",
    "steps_per_epoch": null,
    "warmup_steps": 0,
    "weight_decay": 0.01
    }



Full Model Architecture
++++++++++++++++++++++++
.. code-block:: python

    SentenceTransformer(
    (0): Transformer({'max_seq_length': 512, 'do_lower_case': False}) with Transformer model: MPNetModel 
    (1): Pooling({'word_embedding_dimension': 768, 'pooling_mode_cls_token': False, 'pooling_mode_mean_tokens': True, 'pooling_mode_max_tokens': False, 'pooling_mode_mean_sqrt_len_tokens': False})
    (2): Normalize()
    )

Citing & Authors
++++++++++++++++++

Walsh, HS, & Andrade, SR. "Semantic Search With Sentence-BERT for Design Information Retrieval." Proceedings of the ASME 2022 International Design Engineering Technical Conferences and Computers and Information in Engineering Conference. Volume 2: 42nd Computers and Information in Engineering Conference (CIE). St. Louis, Missouri, USA. August 14–17, 2022. V002T02A066. ASME. https://doi.org/10.1115/DETC2022-89557

