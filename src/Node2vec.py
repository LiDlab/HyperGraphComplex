import networkx as nx
from node2vec import Node2Vec
import pandas as pd
import argparse
import os

def train(args):
    # PPI
    PPI_path = os.path.join(args.data_path, args.species,
                            "IDChange_PPI_dataset/Mann_PPI_Gene.txt")

    graph = pd.read_csv(PPI_path, sep='\t',header=None)
    edgelist = graph.values.tolist()
    G = nx.from_edgelist(edgelist)
    node2vec = Node2Vec(G, dimensions=args.dimensions, walk_length=args.walk_length, num_walks=args.num_walks, p=args.p, q=args.p)

    # Embed nodes
    model = node2vec.fit(window=args.window, min_count=args.min_count,
                         batch_words=args.batch_words)  # Any keywords acceptable by gensim.Word2Vec can be passed, `diemnsions` and `workers` are automatically passed (from the Node2Vec constructor)
    embedding = model.wv.vectors
    embedding = pd.DataFrame(embedding)
    embedding_path = os.path.join(args.data_path, args.species, args.feature_path,
                                        "Mann_Node2vec_embedding.csv")
    embedding.to_csv(embedding_path, index=False)
    print(embedding_path, " Saved!")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    # Global parameters
    parser.add_argument('--species', type=str, default="Saccharomyces_cerevisiae", help="Species to be used in the analysis.")
    parser.add_argument('--data_path', type=str, default="../data", help="Path to the directory containing data.")
    parser.add_argument('--feature_path', type=str, default="protein_feature", help="Path to save the output data.")

    # Node2Vec parameters
    parser.add_argument('--dimensions', type=int, default=64, help="Number of dimensions for node embeddings.")
    parser.add_argument('--walk_length', type=int, default=80, help="Length of each random walk during node2vec training.")
    parser.add_argument('--num_walks', type=int, default=10, help="Number of random walks to perform for each node.")
    parser.add_argument('--p', type=int, default=8, help="Return parameter for node2vec random walks.")
    parser.add_argument('--q', type=int, default=1, help="In-out parameter for node2vec random walks.")
    parser.add_argument('--window', type=int, default=10, help="Maximum distance between the current and predicted node within a random walk.")
    parser.add_argument('--min_count', type=int, default=1, help="Ignores all words with a total frequency lower than this.")
    parser.add_argument('--batch_words', type=int, default=4, help="Number of words to train on in a single batch for node2vec.")

    args = parser.parse_args()
    print(args)
    train(args)


