# Spliceformer: Transformer-Based Splice Site Prediction [![DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.14019451.svg)](https://zenodo.org/badge/DOI/10.5281/zenodo.14019451.svg)

Spliceformer is a machine learning tool that leverages transformers to detect splice sites in pre-mRNA sequences and estimate the impact of genetic variants on splicing by analyzing long nucleotide sequences (up to 45,000 context length).

This repository provides:

* Jupyter Notebooks: Step-by-step examples for training and inference.
* Pre-trained Weights: Ready-to-use models for splice site prediction and variant effect estimation.

For more details, please refer to the [paper](https://www.nature.com/articles/s42003-024-07298-9).

## Pre-trained Model Overview
The pre-trained weights for Spliceformer-45k and SpliceAI-10k provided in this repository were trained on:

* ENSEMBL v87 Annotations: Protein-coding transcripts with splice junctions.
* RNA-Seq Data:
   * Icelandic Cohort: 17,848 whole blood RNA-Seq samples sequenced by deCODE Genetics.
   * GTEx V8: 15,201 RNA-Seq samples across 49 human tissues.

## Setup

```shell
# Step 1: Clone the repository
git clone https://github.com/benniatli/Spliceformer.git
cd Spliceformer

# Step 2: Install required dependencies
# Install dependencies listed in requirements.txt
pip install -r requirements.txt
```

## Colab

Further examples for making predictions with pre-trained weights and predicting the effect of genetic variants are shown in the following colab notebook:

### `spliceformer-usage.ipynb` [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/drive/15dPjh0OqFaGmBUdisxkFEkVeN4KyImt2?usp=sharing).

## Jupyter notebooks
To reproduce the results in the Jupyter notebooks it is neccesary to run them roughly in the following order:

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

## Citing

If you find this work useful in your research, please consider citing: 

```shell
@article{jonsson2024transformers,
  title={Transformers significantly improve splice site prediction},
  author={J{\'o}nsson, Benedikt A and Halld{\'o}rsson, G{\'\i}sli H and {\'A}rdal, Stein{\th}{\'o}r and R{\"o}gnvaldsson, S{\"o}lvi and Einarsson, Ey{\th}{\'o}r and Sulem, Patrick and Gu{\dh}bjartsson, Dan{\'\i}el F and Melsted, P{\'a}ll and Stef{\'a}nsson, K{\'a}ri and {\'U}lfarsson, Magn{\'u}s {\"O}},
  journal={Communications Biology},
  volume={7},
  number={1},
  pages={1--9},
  year={2024},
  publisher={Nature Publishing Group}
}
```
