import networkx as nx
import pandas as pd
import numpy as np

from sentence_transformers import SentenceTransformer, util

def obtain_graph(edge_df):
    
    # Create a new df storing only attributes of nodes
    sourceAtt = edge_df[['sourceRef', 'sourceType', 'sourceName']].rename(columns={'sourceRef': 'id', 'sourceType': 'type', 'sourceName': 'name'})
    targetAtt = edge_df[['targetRef', 'targetType', 'targetName']].rename(columns={'targetRef': 'id', 'targetType': 'type', 'targetName': 'name'})

    # Concatenate source_df and target_df to stack them vertically
    att_df = pd.concat([sourceAtt, targetAtt], ignore_index=True)
    att_df.drop_duplicates(inplace=True)
    att_df.set_index('id', inplace=True)

    # Add a column to store the general type of the node
    att_df['general_type'] = np.where(att_df['type'].str.contains('Event|event'), 'event', 
                                    np.where(att_df['type'].str.contains('Gateway|gateway'), 'gateway', 
                                    np.where(att_df['type'].str.contains('Task|task'), 'task', 'None')))

    # Create a directed graph from the edge dataframe
    G = nx.from_pandas_edgelist(edge_df, source='sourceRef', target='targetRef', edge_attr=True, create_using=nx.DiGraph)
    # Add attributes to the nodes
    G.add_nodes_from((n, dict(d)) for n, d in att_df.iterrows())

    return G

# write a function so that for each node in a graph compute shortest path distance from closest start/end nodes

def find_shortest_path(G):
    start_node = [node for node in G.nodes if G.in_degree(node) == 0]
    end_node = [node for node in G.nodes if G.out_degree(node) == 0]
    
    # check shortest path of each node in G to each start node and take the minimum
    shortest_path_start = {}
    for node in G.nodes:
        # Find the minimum path length from any start node
        shortest_path_start[node] = min(
            [(nx.shortest_path_length(G, start, node)+1) for start in start_node if nx.has_path(G, start, node)],
            default=float('inf')  # Use infinity if no path exists
        )
    
    # check shortest path of each node in G to each end node and take the minimum
    shortest_path_end = {}
    for node in G.nodes:
        # Find the minimum path length to any end node
        shortest_path_end[node] = min(
            [(nx.shortest_path_length(G, node, end)+1) for end in end_node if nx.has_path(G, node, end)], # add 1 to the path length to account for the node itself and avoid prolem with 0 length path 
            default=float('inf')  # Use infinity if no path exists 
        )
    
    return shortest_path_start, shortest_path_end


def compute_shortest_path_distance(sp_G1, sp_G2):
    # Create an empty matrix of shape (G1.number_of_nodes(), G2.number_of_nodes())
    shortest_path_distance = np.zeros((len(sp_G1), len(sp_G2)))

    for i, node_i in enumerate(sp_G1.keys()):
        for j, node_j in enumerate(sp_G2.keys()):
            max_val = max(sp_G1[node_i], sp_G2[node_j])
            shortest_path_distance[i, j] = 1 - abs(sp_G1[node_i] - sp_G2[node_j]) / max_val

    return shortest_path_distance


def get_similarity_matrix(G1, G2, model):
    # create empty matrix of shape (G1.number_of_nodes(), G2.number_of_nodes())
    label_similarity = np.zeros((G1.number_of_nodes(), G2.number_of_nodes()))
    type_similarity = np.zeros((G1.number_of_nodes(), G2.number_of_nodes()))

    for i, node_i in enumerate(G1.nodes):
        for j, node_j in enumerate(G2.nodes):

            # Compute embedding for both nodes labels
            embedding_1 = model.encode(G1.nodes[node_i]['name'], convert_to_tensor=True)
            embedding_2 = model.encode(G2.nodes[node_j]['name'], convert_to_tensor=True)
            
            # Compute cosine similarity
            label_similarity[i,j] = util.pytorch_cos_sim(embedding_1, embedding_2).item()

            # Check if nodes type match and compute node type similarity
            if G1.nodes[node_i]['type'] == G2.nodes[node_j]['type']:
                type_similarity[i,j] = 1
            elif G1.nodes[node_i]['general_type'] == G2.nodes[node_j]['general_type']:
                type_similarity[i,j] = 0.5
                # additional checks for gateway types: if number of incoming/outgoing edges match add 0.25
                if G1.nodes[node_i]['general_type'] == 'gateway' and G1.out_degree(node_i) == G2.out_degree(node_j) and G1.in_degree(node_i) == G2.in_degree(node_j):
                    type_similarity[i,j] += 0.25

            # Check similairty of neighbours
            

            # Check distance from closest start/end nodes
            start_G1, end_G1 = find_shortest_path(G1)
            start_G2, end_G2 = find_shortest_path(G2)

            start_shortest_path_distance = compute_shortest_path_distance(start_G1, start_G2)
            end_shortest_path_distance = compute_shortest_path_distance(end_G1, end_G2)

            # Compute final similairty score
            similarity_matrix = (0.6*label_similarity + 0.2*type_similarity + 0.1*start_shortest_path_distance + 0.1*end_shortest_path_distance)
                
    return label_similarity, type_similarity, start_shortest_path_distance, end_shortest_path_distance, similarity_matrix


def get_similarity_measure(similarity_matrix):
    # extract the max for each row and column and take the mean
    max_row = np.max(similarity_matrix, axis=1)
    max_col = np.max(similarity_matrix, axis=0)
    return np.mean([np.mean(max_row), np.mean(max_col)])