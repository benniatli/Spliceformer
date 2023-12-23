## Splice-site prediction

This repository contains code and jupyter notebooks to train and evaluate deep learning based splice-site prediction models. 

#### Singularity containers (optional)
To start a pytorch singularity container run:

`$ sh Containers/run_pytorch_container.sh`

To start a tensorflow_1/keras singularity container run:

`$ sh Containers/run_tensorflow_1_container.sh`

Then navigate to `$ cd /splice-site-prediction`.

#### Jupyter notebooks
To reproduce the results in the notebooks it is neccesary to run them roughly in the following order:

1. Construct dataset
    * construct_Ensemble_datasets.ipynb

2. Model training and evaluation
    * train_transformer_model.ipynb
    * train_resnet_40k_model.ipynb
    * train_spliceai_10k_model.ipynb
    * train_spliceai_model_gencode_keras.ipynb
    * test_spliceai_model_pretrained.ipynb

3. Analysis
    * get_splice_site_prediction_scores.ipynb
    * spliceAI_vs_transformer_score_comparison.ipynb
    * get_attention_plots.ipynb
    * get_umap_plots.ipynb
    * plot_transformer_training_loss.ipynb

To run step 3 it is only necessary to run train_transformer_model.ipynb and train_spliceai_10k_model.ipynb from step 2.

It is also possible to skip step 2 and use the pre-trained models instead (/Results/PyTorch_Models).
