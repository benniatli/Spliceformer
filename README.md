## Splice-site prediction

This repository contains code and jupyter notebooks to train and evaluate a transformer model with a 45k context.

#### Jupyter notebooks
To reproduce the results in the notebooks it is neccesary to run them roughly in the following order:

1. Construct dataset
    * construct_Ensemble_datasets.ipynb

2. Model training and evaluation
    * train_transformer_model.ipynb
    * train_spliceai_10k_model.ipynb
    * train_spliceai_model_gencode_keras.ipynb
    * test_spliceai_model_pretrained.ipynb
  
3. Fine-tuning on RNA-Seq annotations (only necessary for some of the analysis notebooks)
   * train_transformer_model-fine_tuning-GTEX-ICE-Combined-all.ipynb

4. Analysis
    * get_splice_site_prediction_scores.ipynb
    * spliceAI_vs_transformer_score_comparison.ipynb
    * get_attention_plots.ipynb
    * get_umap_plots.ipynb
    * plot_transformer_training_loss.ipynb
    * novel_splice_transformer.ipynb
    * get_sqtl_delta_for_transformer.ipynb

To run step 4 it is only necessary to run train_transformer_model.ipynb and train_spliceai_10k_model.ipynb from step 2, but some of the notebooks require models fine-tuned on RNA-Seq data.

It is also possible to skip step 2 and 3 and use the pre-trained models instead (/Results/PyTorch_Models).
