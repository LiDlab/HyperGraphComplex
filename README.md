# π-HyperGraphComplex
## Description
π-HyperGraphComplex is a software for predicting protein complexes based on **HyperGraphComplex** algorithm, which integrates high-order topological information from the PPI network and protein sequence features by simultaneously training the encoder and decoder using the HyperGraph Variational Autoencoder (HGVAE). This process generates latent feature vectors for protein complexes. Subsequently, a deep neural network (DNN) is employed to classify candidate protein sets. 
HyperGraphComplex is described in the paper [“Integration of protein sequence and protein–protein interaction data by hypergraph learning to identify novel protein complexes”](https://academic.oup.com/bib/article/25/4/bbae274/7689912) by Simin Xia, Dianke Li, Xinru Deng, et al.

## Dependencies

All dependencies are included `environment.yml` in the method folder.

You could install all dependencies with `conda`:

 ```sh
   conda env create -f environment.yml
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
