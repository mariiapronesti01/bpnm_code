import os
import random
import re

import numpy as np
import pandas as pd
import networkx as nx

from parser_with_lane import get_info_from_bpmn_file


def obtain_graph(edge_df: pd.DataFrame):
    """
    Function that given an edgedf, returns a directed graph with the nodes and their attributes, compatible with the networkx library.
    
    Input:
    - edge_df: a pandas dataframe with the following columns
        - sourceRef, sourceType, sourceName
        - targetRef, targetType, targetName
    
    Output:
    - G: a networkx directed graph with nodes and their attributes
    """
    
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


def load_bpmn_files_from_specific_folder(folder_path: str):
    """
    Function that loads all BPMN files from a specific folder.
    
    Input:
    - folder_path: the path to the folder containing the BPMN files 
    
    Output:
    - bpmn_files: a list of paths to the BPMN files
    """
    
    # List all files in the given folder
    files = []
    for root, _, filenames in os.walk(folder_path):
        for filename in filenames:
            # Join root and filename to get the full file path
            files.append(os.path.join(root, filename))
    bpmn_files = [file for file in files if file.endswith('.bpmn')]
    return bpmn_files


def load_bpmn_files_from_random_folders(base_path: str, subset_size=2):

    """""
    Load all BPMN files from a random subset of folders in the base directory.
    
    Input:
    - base_path: the path to the base directory containing the folders with BPMN files  
    
    Output: 
    - bpmn_files: a list of paths to the BPMN files
    """""
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
        all_files = [file for folder in folders.values() for file in folder]
        bpmn_files = [file for file in all_files if file.endswith('.bpmn')]

    return bpmn_files


def find_shortest_path(G: nx.DiGraph):
    """
    Function that computes the shortest path length from any start node to any node in the graph and from any node to any end node.
    
    Input:
    - G: a networkx directed graph
    
    Output:
    - shortest_path_start: a dictionary with the nodes as keys and the shortest path length from any start node as values
    - shortest_path_end: a dictionary with the nodes as keys and the shortest path length to any end node as values
    """
    
    start_node = [node for node in G.nodes if G.in_degree(node) == 0]
    end_node = [node for node in G.nodes if G.out_degree(node) == 0]
    
    shortest_path_start = {}
    for node in G.nodes:
        # Find the minimum path length from any start node
        shortest_path_start[node] = min(
            [(nx.shortest_path_length(G, start, node)+1) for start in start_node if nx.has_path(G, start, node)],
            default=G.number_of_nodes()  # Changed infinity to number of nodes in the graph -- condition when there is no path from start to node
        )
    
    shortest_path_end = {}
    for node in G.nodes:
        # Find the minimum path length to any end node
        shortest_path_end[node] = min(
            [(nx.shortest_path_length(G, node, end)+1) for end in end_node if nx.has_path(G, node, end)], # add 1 to the path length to account for the node itself and avoid prolem with 0 length path 
            default=G.number_of_nodes()  # Changed infinity to number of nodes in the graph
        )
    
    return shortest_path_start, shortest_path_end


def read_file(file: str):
    """
    Function that reads a file and returns its content.
    
    Input:
    - file: a string containing the path to the file
    """
    with open(file, 'r') as f:
        return f.read()


def build_process_info_dict(all_files: list, model):
    """
    Function that builds a dictionary containing the information extracted from the BPMN files.
    
    Input:
    - all_files: a list of paths to the BPMN files
    - model: a sentence transformer model
    
    Output:
    - files_info: a dictionary containing the following information:
        - edge_df: a pandas dataframe with the edge information
        - lane_info: a dictionary containing the lane information
        - G: a networkx directed graph
        - start_shortest_path: a dictionary with the shortest path length from any start node
        - end_shortest_path: a dictionary with the shortest path length to any end node
        - embeddings: a dictionary containing the embeddings of the nodes
        - degree: a dictionary containing the in and out degree of the nodes
    """
    
    files_info = {}
    for file in all_files:
        edge_df, lane_info = get_info_from_bpmn_file(file)  # Unpack the tuple
        graph = obtain_graph(edge_df)
        shortest_path_start, shortest_path_end = find_shortest_path(graph)
        embeddings = {node : model.encode(graph.nodes[node]['name'], convert_to_tensor=True) for node in graph.nodes}
        degree = {node: (graph.in_degree(node), graph.out_degree(node)) for node in graph.nodes}
        files_info[file] = {
            'edge_df': edge_df,   # Assign to 'edge_df' key
            'lane_info': lane_info, # Assign to 'lane_info' key 
            'G' : graph, # Assign to 'graph' key
            'start_shortest_path' : shortest_path_start, # Assign to 'startShortestPath' key
            'end_shortest_path' : shortest_path_end, # Assign to 'endShortestPath' key
            'embeddings' : embeddings, # Assign to 'embeddings' key
            'degree' : degree # Assign to 'degree' key
        }
    return files_info


def getMER_fromBPMN(file: str):
    """
    Function that writes a pseudo MERMAID file for a given BPMN file.
    """

    df, lane_info = get_info_from_bpmn_file(file)
    #df = df.apply(add_brackets, axis=1)

    output = file.replace('.bpmn', '.txt')
    
    with open(output, 'w') as f:  # Open the file for writing once
        # Write connections between nodes from different lanes (outside of subgraphs)
        for _, row in df.iterrows():
            f.write(
                f"{row['sourceType']}//{row['sourceRef']}({row['sourceName']})"   # changed brackets to be () for all nodes
                f"-->|{row['id']}({row['name']})|"
                f"{row['targetType']}//{row['targetRef']}({row['targetName']})\n"
            )
        if lane_info:
        # Write subgraphs for each lane
            for lane_id, lane_data in lane_info.items():
                # Write the subgraph label for the current lane
                f.write(f"subgraph {lane_id}({lane_data['name']})\n")

                # Write all nodes associated with the current lane
                current_lane_nodes = set(lane_data['nodes'])
                for node in current_lane_nodes:
                    f.write(f"  {node}\n")  # Write each node in the subgraph

                # Close the subgraph for the current lane
                f.write("end\n\n")
                
                

def getProcessInfo_fromMer(mer_file: str):
    """
    Function that extracts the edgeDF and lane info from a MERMAID file.
    
    Input:
    - mer_file: a string containing the MERMAID file
    
    Output:
    - df: a pandas dataframe with the edge information
    - lane_info: a dictionary containing lane information
    """
    
    # Initialize a list to store the rows and a dictionary for lane info
    rows = []
    lane_info = {}

    # Split the input string by newlines to get individual entries
    bpmn_list = mer_file.strip().split('\n')

    edge_pattern = r'(\w+//([\w-]+)\((.*?)\))-->\|([\w-]+)\((.*?)\)\|(\w+//([\w-]+)\((.*?)\))'
    lane_pattern = r'subgraph\s+([\w-]+)\((.*?)\)'

    for entry in bpmn_list:
        edge_match = re.match(edge_pattern, entry)
        lane_match = re.match(lane_pattern, entry)
        
        if edge_match:
            source_node_full = edge_match.group(1)
            edge_id = edge_match.group(4)
            edge_name = edge_match.group(5)
            target_node_full = edge_match.group(6)

                # Further split source and target nodes
            source_node_type, source_node_id, source_node_name = (
                source_node_full.split('//', 1)[0],
                source_node_full.split('//', 1)[1].rsplit('(', 1)[0],
                source_node_full.split('(', 1)[1][:-1]  # Removing the closing bracket
                )

            target_node_type, target_node_id, target_node_name = (
                target_node_full.split('//', 1)[0],
                target_node_full.split('//', 1)[1].rsplit('(', 1)[0],
                target_node_full.split('(', 1)[1][:-1]  # Removing the closing brace
                )

                # Append the extracted data to rows
            rows.append([
                edge_id,
                edge_name,
                source_node_type,
                source_node_id,
                target_node_type,
                target_node_id,
                source_node_name,
                target_node_name
            ])
        elif lane_match:
            lane_id = lane_match.group(1)
            lane_name = lane_match.group(2)
            lane_nodes = []

            # Collect node identifiers in the lane
            for node_entry in bpmn_list[bpmn_list.index(entry) + 1:]:
                if node_entry.strip() == 'end':
                    break  # Stop if we reach the end of the lane definition
                lane_nodes.append(node_entry.strip())

            # Store each lane's information in a dictionary, keyed by lane_id
            lane_info[lane_id] = {
                'name': lane_name,
                'nodes': lane_nodes
            }

    # Create a DataFrame from the rows
    columns = ['edgeID', 'edgeName', 'sourceType', 'sourceRef', 
               'targetType', 'targetRef', 'sourceName', 'targetName']
    
    df = pd.DataFrame(rows, columns=columns)
    return df, lane_info
