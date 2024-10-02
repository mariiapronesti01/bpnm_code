import xml.etree.ElementTree as ET
import pandas as pd
import re

def flatten(xss):
    return [x for xs in xss for x in xs]

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


def extract_lane(process, ns):
    lane_info = {}
    for lane in process.findall(f'{ns}laneSet/{ns}lane'):
            lane_id = lane.attrib['id']
            lane_name = lane.attrib.get('name', 'Unnamed')  # Provide default if 'name' is missing
            lane_nodes = [flow_node_ref.text for flow_node_ref in lane.findall(f'{ns}flowNodeRef')]
            
            # Store each lane's information in a dictionary, keyed by lane_id
            lane_info[lane_id] = {
                'name': lane_name,
                'nodes': lane_nodes
            }
    
    return lane_info


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

    all_node_type = []
    all_attributes = []
    lane_info = {}
            
    for process in processes:
                node_type, attributes = extract_process_info(process)
                all_node_type.append(node_type)
                all_attributes.append(attributes)
                lane_info.update(extract_lane(process, ns))

    node_type = flatten(all_node_type)
    attributes = flatten(all_attributes)

    print(f'Detected {len(processes)} processes in the BPMN file')

    return node_type, attributes, lane_info


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
    node_type, attributes, _ = parse_bpmn(ns, file)
    df = get_df(attributes, node_type, complete)
    if brackets:
        df = df.apply(add_brackets, axis=1)
    return df


def write_mer_file(df, output, lane_info):
    
    with open(output, 'w') as f:  # Open the file for writing once
        # Write connections between nodes from different lanes (outside of subgraphs)
        for _, row in df.iterrows():
            f.write(
                f"{row['sourceRef']}{row['sourceOpen']}{row['sourceName']}{row['sourceClose']} "
                f"--> | {row['name']} | "
                f"{row['targetRef']}{row['targetOpen']}{row['targetName']}{row['targetClose']}\n"
            )
        if lane_info:
        # Write subgraphs for each lane
            for _, lane_data in lane_info.items():
                # Write the subgraph label for the current lane
                f.write(f"subgraph {lane_data['name']}\n")

                # Write all nodes associated with the current lane
                current_lane_nodes = set(lane_data['nodes'])
                for node in current_lane_nodes:
                    f.write(f"  {node}\n")  # Write each node in the subgraph

                # Close the subgraph for the current lane
                f.write("end\n\n")

