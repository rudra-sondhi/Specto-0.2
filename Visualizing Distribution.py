import yaml
from collections import Counter, defaultdict
import matplotlib.pyplot as plt
import seaborn as sns
import networkx as nx

import pandas as pd

# Convert YAML data to DataFrame
def convert_to_dataframe(yaml_data):
    df = pd.DataFrame(yaml_data).T  # Transpose to turn keys into rows
    return df

# One-hot Encode Functional Groups
def one_hot_encode_functional_groups(df):
    # Explode the 'Functional Groups' column into separate rows
    df_exploded = df['Functional Groups'].explode()
    # Perform one-hot encoding
    one_hot = pd.get_dummies(df_exploded).groupby(level=0).sum()
    # Join back with the original dataframe
    df = df.join(one_hot)
    return df



# Function to Load Data and Count Functional Groups
def load_data_and_count_fg(yaml_file):
    with open(yaml_file, 'r') as file:
        yaml_data = yaml.safe_load(file)
    functional_groups = [fg for item in yaml_data.values() for fg in item.get('Functional Groups', [])]
    return Counter(functional_groups), yaml_data

# Function to Compute Co-occurrences
def compute_co_occurrence(yaml_data):
    co_occurrence = defaultdict(lambda: defaultdict(int))
    for entry in yaml_data.values():
        fg_list = entry.get('Functional Groups', [])
        for fg1 in fg_list:
            for fg2 in fg_list:
                if fg1 != fg2:
                    co_occurrence[fg1][fg2] += 1
    return co_occurrence

# Function to Create and Display Bar Plot
def display_bar_plot(fg_counts):
    fg_names = list(fg_counts.keys())
    counts = list(fg_counts.values())
    plt.figure(figsize=(10, 6))
    plt.bar(fg_names, counts, color='skyblue')
    plt.xlabel('Functional Groups')
    plt.ylabel('Occurrences')
    plt.title('Functional Group Occurrences in the Dataset')
    plt.xticks(rotation=45, ha="right")
    plt.tight_layout()
    plt.show()

def display_normalized_bar_plot(fg_counts):
    fg_names = list(fg_counts.keys())
    total_count = sum(fg_counts.values())
    normalized_counts = [count / total_count for count in fg_counts.values()]

    plt.figure(figsize=(10, 6))
    plt.bar(fg_names, normalized_counts, color='skyblue')
    plt.xlabel('Functional Groups')
    plt.ylabel('Normalized Occurrences')
    plt.title('Normalized Functional Group Occurrences in the Dataset')
    plt.xticks(rotation=45, ha="right")
    plt.tight_layout()
    plt.show()

# Function to Display Heatmap
def display_co_occurrence_heatmap(co_occurrence):
    fg_names = list(co_occurrence.keys())
    co_occurrence_matrix = [[co_occurrence[fg1].get(fg2, 0) for fg2 in fg_names] for fg1 in fg_names]
    plt.figure(figsize=(12, 8))
    sns.heatmap(co_occurrence_matrix, annot=True, fmt='d', xticklabels=fg_names, yticklabels=fg_names)
    plt.title('Co-occurrence of Functional Groups')
    plt.show()
    return co_occurrence_matrix, fg_names

# Function to Create and Display Network Graph
def display_network_graph(co_occurrence):
    G = nx.Graph()
    for fg1 in co_occurrence:
        for fg2 in co_occurrence[fg1]:
            if co_occurrence[fg1][fg2] > 0:
                G.add_edge(fg1, fg2, weight=co_occurrence[fg1][fg2])
    plt.figure(figsize=(12, 10))
    pos = nx.spring_layout(G, k=0.1)
    nx.draw(G, pos, with_labels=True, node_color='skyblue', edge_color='gray', width=1, edge_cmap=plt.cm.Blues)
    plt.title('Network Graph of Functional Group Co-occurrences')
    plt.show()

from itertools import combinations

# Modified Function to Compute Triple Co-occurrence
def compute_triple_co_occurrence(yaml_data):
    triple_co_occurrence = defaultdict(lambda: defaultdict(lambda: defaultdict(int)))
    for entry in yaml_data.values():
        fg_list = entry.get('Functional Groups', [])
        for fg1, fg2, fg3 in combinations(fg_list, 3):
            triple_co_occurrence[fg1][fg2][fg3] += 1
            triple_co_occurrence[fg1][fg3][fg2] += 1
            triple_co_occurrence[fg2][fg1][fg3] += 1
            triple_co_occurrence[fg2][fg3][fg1] += 1
            triple_co_occurrence[fg3][fg1][fg2] += 1
            triple_co_occurrence[fg3][fg2][fg1] += 1
    return triple_co_occurrence

# Function to Find and Display Most Connected Triples
def display_top_triples(triple_co_occurrence, top_n=10):
    triple_counts = []
    for fg1 in triple_co_occurrence:
        for fg2 in triple_co_occurrence[fg1]:
            for fg3 in triple_co_occurrence[fg1][fg2]:
                count = triple_co_occurrence[fg1][fg2][fg3]
                triple_counts.append((count, (fg1, fg2, fg3)))
    top_triples = sorted(triple_counts, reverse=True)[:top_n]
    print(f"Top {top_n} Most Connected Functional Group Triples:")
    for count, triple in top_triples:
        print(f"{triple}: {count} times")

# Function to Display Top 10 Co-occurring Pairs and Total Molecules
def display_top_pairs_and_total_molecules(co_occurrence, yaml_data, top_n=10):
    pair_counts = []
    for fg1 in co_occurrence:
        for fg2 in co_occurrence[fg1]:
            count = co_occurrence[fg1][fg2]
            pair_counts.append((count, (fg1, fg2)))
    top_pairs = sorted(pair_counts, reverse=True)[:top_n]
    total_molecules = len(yaml_data)
    print(f"Top {top_n} Most Co-occurring Functional Group Pairs:")
    for count, pair in top_pairs:
        print(f"{pair}: {count} times")
    print(f"\nTotal molecules in the database: {total_molecules}")

def display_bar_plot_with_labels(fg_counts):
    fg_names = list(fg_counts.keys())
    counts = list(fg_counts.values())
    
    plt.figure(figsize=(10, 6))
    bars = plt.bar(fg_names, counts, color='skyblue')
    plt.xlabel('Functional Groups')
    plt.ylabel('Occurrences')
    plt.title('Functional Group Occurrences in the Dataset')

    # Adding labels on top of each bar
    for bar in bars:
        yval = bar.get_height()
        plt.text(bar.get_x() + bar.get_width()/2, yval, int(yval), ha='center', va='bottom')

    plt.xticks(rotation=45, ha="right")
    plt.tight_layout()
    plt.show()





#Example of how to call these functions (commented out to prevent execution)
file_path = '/Users/rudrasondhi/Desktop/Specto/Specto/Data/IR_Spectra/all_data_SMARTS.yaml'
fg_counts, yaml_data = load_data_and_count_fg(file_path)

co_occurrence = compute_co_occurrence(yaml_data)

# Convert and preprocess data
df = convert_to_dataframe(yaml_data)
print(len(df))
df_preprocessed = one_hot_encode_functional_groups(df)
"""
display_bar_plot(fg_counts)
display_co_occurrence_heatmap(co_occurrence)
display_network_graph(co_occurrence)
display_normalized_bar_plot(fg_counts)

triple_co_occurrence = compute_triple_co_occurrence(yaml_data)
display_top_triples(triple_co_occurrence)
display_top_pairs_and_total_molecules(co_occurrence, yaml_data)
display_bar_plot_with_labels(fg_counts)
"""






