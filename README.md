<h1 align="center">π-HyperGraphComplex</h1>

## Description
 
  ![HyperGraphComplex Architecture](data/Figure 2.png)

π-HyperGraphComplex is a software for predicting protein complexes based on **HyperGraphComplex** algorithm, which integrates high-order topological information from the PPI network and protein sequence features by simultaneously training the encoder and decoder using the HyperGraph Variational Autoencoder (HGVAE). This process generates latent feature vectors for protein complexes. Subsequently, a deep neural network (DNN) is employed to classify candidate protein sets.

HyperGraphComplex is described in the paper [“Integration of protein sequence and protein–protein interaction data by hypergraph learning to identify novel protein complexes”](https://academic.oup.com/bib/article/25/4/bbae274/7689912) by Simin Xia, Dianke Li, Xinru Deng, et al.

## Dependencies

All dependencies are listed in the `environment.yml` file located in the `method` folder.

To install all dependencies using `conda`, run:

 ```sh
   conda env create -f environment.yml
 ```

## Usage
### Training
Train the HyperGraphComplex model based on Mann PPI data. Navigate to the `src` directory and then execute the training script:
 ```sh
  python main.py
 ```

### Prediction
Predict protein complexes using the HyperGraphComplex model. Navigate to the `src` directory and then execute the prediction script:
 ```sh
  python Predicting_protein_complex.py
 ```
