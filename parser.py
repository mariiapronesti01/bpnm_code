import xml.etree.ElementTree as ET
import pandas as pd
import re

def extract_process_info(processes):
    
    """
    Helper function to extract the node type and attributes of the process elements

    """
    node_type = []
    attributes = []
    for child in processes:
        node_type.append(re.sub(r'{.*}', '', child.tag))
        attributes.append(child.attrib)
    return node_type, attributes


def parse_bpmn(ns: str, file: str):

    """
    Function to parse the BPMN file and extract the process(es) elements

    Parameters:
    - ns: namespace of the BPMN file
    - file: path to the BPMN file

    Returns:
    - node_type: list of node types
    - attributes: list of dictionaries containing the attributes of the process elements

    """
   # Parse the XML file
    tree = ET.parse(file)
    root = tree.getroot()

    # Extract the process elements
    processes = root.findall(f'{ns}process')

    # Check if there are multiple processes
    if len(processes) > 1:
        all_node_type = []
        all_attributes = []
        
        for process in processes:
            node_type, attributes = extract_process_info(process)
            all_node_type.append(node_type)
            all_attributes.append(attributes)

        print(f'Detected {len(processes)} processes in the BPMN file')
        return all_node_type, all_attributes

    # Handle the case with a single process
    else:
        process = processes[0]
        node_type, attributes = extract_process_info(process)
        
        print(f'Detected 1 process in the BPMN file')
        return node_type, attributes



def get_df(attributes: list, node_type: list, complete=False):

    """
    Function to create a DataFrame from the attributes of the process elements

    Parameters:
    - attributes: list of dictionaries containing the attributes of the process elements
    - node_type: list of node types
    - complete: boolean to return the complete DataFrame or only the sequenceFlow elements

    Returns:
    - df: DataFrame containing the attributes of the process elements

    """
    
    df = pd.DataFrame(attributes)
    df['node_type'] = node_type

    # map id to node type and name
    id_to_node_type = df.set_index('id')['node_type'].to_dict()
    id_to_name = df.set_index('id')['name'].to_dict()

    # add source type and target type columns
    df['sourceType']=df['sourceRef'].map(id_to_node_type)
    df['targetType']=df['targetRef'].map(id_to_node_type)

    # add source name and target name columns
    df['sourceName']=df['sourceRef'].map(id_to_name)
    df['targetName']=df['targetRef'].map(id_to_name)

    if complete:
        return df
    
    return df[df['node_type']=='sequenceFlow']


def add_brackets(row):
        
        """
        Helper function to add brackets to the source and target names based on their type. Needed for mermaid representation

        """
        # For 'sourceRef' column (open/close symbols)
        if re.search('Event', row['sourceType'], re.IGNORECASE):
            row['sourceOpen'], row['sourceClose'] = '(( ', ' ))'
        elif re.search('task', row['sourceType'], re.IGNORECASE):    
            row['sourceOpen'], row['sourceClose'] = '[ ', ' ]'
        elif re.search('Gateway', row['sourceType'], re.IGNORECASE):
            row['sourceOpen'], row['sourceClose'] = '{ ', ' }'

        # For 'targetRef' column (open/close symbols)
        if re.search('Event', row['targetType'], re.IGNORECASE):
            row['targetOpen'], row['targetClose'] = '(( ', ' ))'
        elif re.search('task', row['targetType'], re.IGNORECASE):
            row['targetOpen'], row['targetClose'] = '[ ', ' ]'
        elif re.search('Gateway', row['targetType'], re.IGNORECASE):
            row['targetOpen'], row['targetClose'] = '{ ', ' }'

        return row

def write_mer_file(df, output: str):

    """
    Function to write the formatted output to a file

    Parameters:
    - df: DataFrame containing the formatted output
    - output: path to the output file

    """
    with open(output, 'w') as f:
        for _, df in enumerate(df):
                # Write the formatted output for each row in the DataFrame to the file
                for _, row in df.iterrows():
                    f.write(
                        f"{row['sourceRef']}{row['sourceOpen']}{row['sourceName']}{row['sourceClose']} --> | {row['name']} | {row['targetRef']}{row['targetOpen']}{row['targetName']}{row['targetClose']}\n"
                    )

def get_edge_df_from_bpmn(file:str, ns='{http://www.omg.org/spec/BPMN/20100524/MODEL}', brackets=False, complete=False):

    """
    Function to extract the edges from the BPMN file and return a DataFrame

    Parameters:
    - ns: namespace of the BPMN file
    - file: path to the BPMN file
    - brackets: boolean to add brackets to the source and target names

    Returns:
    - df: DataFrame containing the edges of the process elements

    """
    node_type, attributes = parse_bpmn(ns, file)
    process_count = sum(isinstance(i, list) for i in node_type) or 1   # count number of sublist in tasks, if None then # of processes = 1 else # of processes = # of sublists 
    node_type = node_type if process_count > 1 else [node_type]
    attributes = attributes if process_count > 1 else [attributes]

    dfs = []
    for i in range(process_count):
        df = get_df(attributes[i], node_type[i], complete)
        if brackets:
            df = df.apply(add_brackets, axis=1)
        dfs.append(df)
    return dfs