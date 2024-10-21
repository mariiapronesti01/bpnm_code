import numpy as np
import networkx as nx

import seaborn as sns
import matplotlib.pyplot as plt

from sentence_transformers import SentenceTransformer, util

def get_TypeSimilarityMatrix(G1: nx.DiGraph, G2: nx.DiGraph, degree_G1: dict, degree_G2: dict):
    """""
    Compute the similarity between the types of two nodes of two processes.
    
    Input:
    - G1: a networkx directed graph representing the first process
    - G2: a networkx directed graph representing the second process
    - degree_G1: a dictionary containing the in and out degree of the nodes of the first process
    - degree_G2: a dictionary containing the in and out degree of the nodes of the second process
    
    Output:
    - type_similarity: a numpy array of shape ((G1.number_of_nodes(), G2.number_of_nodes())) containing the similarity between the types of the nodes of the two processes
    """""
    type_similarity = np.zeros((len(G1.nodes), len(G2.nodes)))

    for i, node_i in enumerate(G1.nodes):
        node_i_type = G1.nodes[node_i]['type']
        node_i_general_type = G1.nodes[node_i]['general_type']
        in_degree_i, out_degree_i = degree_G1[node_i]

        for j, node_j in enumerate(G2.nodes):
            # Compute type similarity
            if node_i_type == G2.nodes[node_j]['type']:
                type_similarity[i, j] = 1
            elif node_i_general_type == G2.nodes[node_j]['general_type']:
                type_similarity[i, j] = 0.5
                # Additional gateway checks
                in_degree_j, out_degree_j = degree_G2[node_j]
                if node_i_general_type == 'gateway' and in_degree_i == in_degree_j and out_degree_i == out_degree_j:
                    type_similarity[i, j] += 0.25
                
    return type_similarity


def get_LabelSimilarityMatrix(embedding1: dict, embedding2: dict):
    """
    Compute the similarity between the labels of two nodes of two processes.
    
    Input:
    - embedding1: a dictionary containing the embeddings of the nodes of the first process
    - embedding2: a dictionary containing the embeddings of the nodes of the second process
    
    Output:
    - label_similarity_matrix: a numpy array of shape ((len(embedding1), len(embedding2))) containing the cosine similarity between the labels of the nodes of the two processes
    """
    label_similarity_matrix = np.zeros((len(embedding1), len(embedding2)))

    for i, node_i in enumerate(embedding1):
        embedding_i = embedding1[node_i]
        for j, node_j in enumerate(embedding2):
            embedding_j = embedding2[node_j]
            label_similarity_matrix[i, j] = util.pytorch_cos_sim(embedding_i, embedding_j).item()
    return label_similarity_matrix


def get_NeighbourSimilarityMatrix(G1: nx.DiGraph, G2: nx.DiGraph, label_similarity_matrix: np.array, type_similarity_matrix: np.array):
    """
    
    Compute the similarity between the neighbours of each node of two processes.
    
    Input:
    - G1: a networkx directed graph representing the first process
    - G2: a networkx directed graph representing the second process
    - label_similarity_matrix: a numpy array of shape ((G1.number_of_nodes(), G2.number_of_nodes())) containing the cosine similarity between the labels of the nodes of the two processes
    - type_similarity_matrix: a numpy array of shape ((G1.number_of_nodes(), G2.number_of_nodes())) containing the similarity between the types of the nodes of the two processes
    
    Output:
    - neighbour_similarity_matrix: a numpy array of shape ((G1.number_of_nodes(), G2.number_of_nodes())) containing the similarity score between the neighbours of the nodes of the two processes
    """
    # create mapping between node and index
    node_to_index_G1 = {node: i for i, node in enumerate(G1.nodes)} 
    node_to_index_G2 = {node: i for i, node in enumerate(G2.nodes)}

    # Compute the similarity between the neighbours of each node
    neighbour_similarity_matrix = np.zeros((len(G1.nodes), len(G2.nodes)))

    for i, node_i in enumerate(G1.nodes):
        for j, node_j in enumerate(G2.nodes):
            # Get the neighbours of the current nodes
            neighbours_i = list(G1.successors(node_i)) + list(G1.predecessors(node_i))
            neighbours_j = list(G2.successors(node_j)) + list(G2.predecessors(node_j))

            # recover label similarity from label similarity matrix
            label_similarity = 0
            type_similarity = 0
            for neighbour_i in neighbours_i:
                for neighbour_j in neighbours_j:
                    label_similarity += label_similarity_matrix[node_to_index_G1[neighbour_i], node_to_index_G2[neighbour_j]]   # label_similarity[index_neighbour_i, index_neighbour_j]
                    type_similarity += type_similarity_matrix[node_to_index_G1[neighbour_i], node_to_index_G2[neighbour_j]]
            neighbour_similarity_matrix[i, j] = (label_similarity + type_similarity)/(len(neighbours_i) * len(neighbours_j))
    return neighbour_similarity_matrix


def get_ShortestPathDistanceMatrix(shortest_path_G1: dict, shortest_path_G2: dict):
    """
    Compute the similarity between the shortest path of each node of two processes.
    
    Input:
    - shortest_path_G1: a dictionary with the nodes as keys and the shortest path length from any start/end node as values for the first process
    - shortest_path_G2: a dictionary with the nodes as keys and the shortest path length from any start/end node as values for the second process
    
    Output:
    - shortest_path_distance: a numpy array of shape ((G1.number_of_nodes(), G2.number_of_nodes())) containing the similarity score between the shortest path of the nodes of the two processes
    """
    
    # Convert the shortest path dicts to numpy arrays for faster operations
    sp_G1_values = np.array(list(shortest_path_G1.values()))
    sp_G2_values = np.array(list(shortest_path_G2.values()))

    # Create a grid of differences and maximum values
    diff = np.abs(sp_G1_values[:, np.newaxis] - sp_G2_values)
    max_val = np.maximum(sp_G1_values[:, np.newaxis], sp_G2_values)

    shortest_path_distance = 1 - (diff / max_val)
    return shortest_path_distance


def get_2ProcessesSimilarity(info_process1: dict, info_process2: dict, return_matrix=False):
    """
    Compute the similarity score between two processes.
    
    Input:
    - info_process1: a dictionary containing the information of the first process
    - info_process2: a dictionary containing the information of the second process
    
    Output:
    - similarity_score: a float value representing the similarity score between the two processes
    """
    # Calculate the different similarity metrics
    type_similarity = get_TypeSimilarityMatrix(info_process1['G'], info_process2['G'], info_process1['degree'], info_process2['degree'])
    label_similarity = get_LabelSimilarityMatrix(info_process1['embeddings'], info_process2['embeddings'])
    neighbor_similarity = get_NeighbourSimilarityMatrix(info_process1['G'], info_process2['G'], label_similarity, type_similarity)
    start_shortest_path_distance = get_ShortestPathDistanceMatrix(info_process1['start_shortest_path'], info_process2['start_shortest_path'])
    end_shortest_path_distance = get_ShortestPathDistanceMatrix(info_process1['end_shortest_path'], info_process2['end_shortest_path'])

    
    # Combine the similarities with the given weights
    similarity_matrix = (
        0.20 * label_similarity + 
        0.20 * type_similarity + 
        0.20 * start_shortest_path_distance + 
        0.20 * end_shortest_path_distance +
        0.20 * neighbor_similarity
    )

    max_row_mean = np.max(similarity_matrix, axis=1).mean()
    max_col_mean = np.max(similarity_matrix, axis=0).mean()
    similarity_score = (max_row_mean + max_col_mean) / 2

    if return_matrix:
        return similarity_matrix, similarity_score
    else:
        return similarity_score


def get_AllFilesSimilarityMatrix(files_info: dict):
    """
    Compute the similarity matrix between a set of processes.
    
    Input:
    - files_info: a dictionary containing the information of the processes
    
    Output:
    - all_files_similarity_matrix: a numpy array of shape ((num_files, num_files)) containing the similarity score between pairs of processes
    """
    # Initialize the similarity matrix with zeros
    num_files = len(files_info)
    all_files_similarity_matrix = np.zeros((num_files, num_files))
    
    # Get the list of keys in files_info to index into the dictionary
    file_keys = list(files_info.keys())
    
    # Loop over the file keys to compare each file with every other file
    for i in range(num_files):
        for j in range(i, num_files):
            file_i = files_info[file_keys[i]]
            file_j = files_info[file_keys[j]] 
            
            if i == j:
                # Set the diagonal value to 1 (self-similarity)
                all_files_similarity_matrix[i, i] = 1
            else:
                # Compute the overall similarity and fill the symmetric matrix
                all_files_similarity_matrix[i, j] = get_2ProcessesSimilarity(file_i, file_j)
                all_files_similarity_matrix[j, i] = all_files_similarity_matrix[i, j]

    return all_files_similarity_matrix


def plot_AllFilesSimilarityMatrix(all_files_similarity_matrix: np.array, labels: list):
    """
    Plot the similarity matrix between a set of processes.
    
    Input:
    - all_files_similarity_matrix: a numpy array of shape ((num_files, num_files)) containing the similarity score between pairs of processes
    - labels: a list of labels for the processes
    """
    # Plot the similarity matrix
    plt.figure(figsize=(20, 10))
    sns.heatmap(all_files_similarity_matrix, annot=True, fmt=".2f", cmap='YlGn', xticklabels=labels, yticklabels=labels)
    plt.title("Similarity Matrix")
    plt.show()