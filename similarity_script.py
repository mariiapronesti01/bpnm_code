#!/usr/bin/env python3

import os
import random
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt

from similarity_utils import obtain_graph, get_similarity_measure, get_similarity_matrix
from parser_with_lane import get_edge_df_from_bpmn
from sentence_transformers import SentenceTransformer, util

model = SentenceTransformer("sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2")

def check_file_similarity(file1, file2, model, verbose=False):
    edge_df1 = get_edge_df_from_bpmn(file1)
    edge_df2 = get_edge_df_from_bpmn(file2)

    G1 = obtain_graph(edge_df1)
    G2 = obtain_graph(edge_df2)

    if verbose:
        print(f"Graph 1 has:", G1.number_of_nodes(), "nodes and", G1.number_of_edges(), "edges")
        print(f"Graph 2 has:", G2.number_of_nodes(), "nodes and", G2.number_of_edges(), "edges")

    _, _, _, _, weighted_similarity_matrix, unweighted_similarity_matrix = get_similarity_matrix(G1, G2, model)
    return get_similarity_measure(weighted_similarity_matrix), get_similarity_measure(unweighted_similarity_matrix)

def load_files_from_folders(base_path, subset_size=4):
    folders = {}
    # List all folders in the base directory
    all_folders = [folder_name for folder_name in os.listdir(base_path) if os.path.isdir(os.path.join(base_path, folder_name))]
    
    # Select a random subset of 12 folders
    selected_folders = random.sample(all_folders, min(subset_size, len(all_folders)))

    # Load files from the selected folders
    for folder_name in selected_folders:
        folder_path = os.path.join(base_path, folder_name)
        files = [os.path.join(folder_path, f) for f in os.listdir(folder_path) if os.path.isfile(os.path.join(folder_path, f))]
        folders[folder_name] = files

    return folders

def compute_similarity_matrix(all_files, model, verbose=False):
    n = len(all_files)
    w_similarity_matrix = np.zeros((n, n))
    unw_similarity_matrix = np.zeros((n, n))

    for i in range(n):
        for j in range(i, n):
            # Compute similarity once for (i, j) and reuse it for (j, i)
            if verbose:
                print(f"Computing similarity between {all_files[i]} and {all_files[j]}")

            w_similarity, unw_similairy = check_file_similarity(all_files[i], all_files[j], model)
            w_similarity_matrix[i, j] = w_similarity
            w_similarity_matrix[j, i] = w_similarity  # Exploit symmetry
            unw_similarity_matrix[i, j] = unw_similairy
            unw_similarity_matrix[j, i] = unw_similairy

    return w_similarity_matrix, unw_similarity_matrix

def main():
    folders = load_files_from_folders("cleaned_data", subset_size=4)
    print(folders)
    all_files = [file for folder in folders.values() for file in folder]
    all_files = [file for file in all_files if file.endswith('.bpmn')]

    w_similarity_matrix, unw_similarity_matrix = compute_similarity_matrix(all_files, model)

    # save the similarity matrices
    np.save("weighted_similarity_matrix.npy", w_similarity_matrix)
    np.save("unweighted_similarity_matrix.npy", unw_similarity_matrix)

    # save the plot of the similariy matrix toa file
    plt.figure(figsize=(20, 10))  # Adjust height for better fit (20, 10) for side-by-side comparison

    # First subplot for weighted similarity matrix
    plt.subplot(1, 2, 1)  # 1 row, 2 columns, 1st plot
    plt.title("Weighted Similarity Matrix")
    sns.heatmap(w_similarity_matrix, annot=True, xticklabels=all_files, yticklabels=all_files, cmap='YlGn')

    # Second subplot for unweighted similarity matrix
    plt.subplot(1, 2, 2)  # 1 row, 2 columns, 2nd plot
    plt.title("Unweighted Similarity Matrix")
    sns.heatmap(unw_similarity_matrix, annot=True, xticklabels=all_files, yticklabels=all_files, cmap='YlGn')

    plt.tight_layout()  # Adjust layout to prevent overlapping of plots
    plt.savefig("similarity_matrix.png")



if __name__ == "__main__":
    main()
    