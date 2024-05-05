# HyperGraphComplex
## Description
This is a protein complex prediction model based on hypergraph representation learning. HyperGraphComplex integrates high-order topological information from the protein-protein interaction (PPI) network and protein sequence features by simultaneously training the encoder and decoder using the HyperGraphVariational Autoencoder (HGVAE). This process generates latent feature vectors for protein complexes. Subsequently, a deep neural network (DNN) is employed to classify candidate protein sets. 

## Dependencies

All dependencies are included `environment.yaml` in the method folder.

You could install all dependencies with `conda`:

 ```sh
   conda env create -f environment.yaml
 ```

## Usage
Traning HyperGraphComplex  base on Mann PPI . 
Enter the method folder：
 ```sh
  python main.py
 ```
Predicting protein complex by HyperGraphComplex.  Enter the method folder：
 ```sh
  python Predicting_protein_complex.py
 ```
