import pandas as pd
import networkx as nx

# Load dataset
csv_filename = "waste_classification_data.csv"
df = pd.read_csv(csv_filename)

# Initialize a graph
G = nx.Graph()

# Add nodes (categories: Waste Type, Source, Disposal Method, Area Name, Collection Frequency)
categories = ["Waste Type", "Source", "Disposal Method", "Area Name", "Collection Frequency"]
for category in categories:
    G.add_nodes_from(df[category].unique(), type=category)

# Add edges with weights based on waste quantity or occurrence count
for _, row in df.iterrows():
    # Waste Type to Source (sum of waste quantity)
    G.add_edge(row["Waste Type"], row["Source"], weight=row["Waste Quantity (kg)"])

    # Waste Type to Disposal Method (sum of waste quantity)
    G.add_edge(row["Waste Type"], row["Disposal Method"], weight=row["Waste Quantity (kg)"])

    # Area to Waste Type (sum of waste quantity)
    G.add_edge(row["Area Name"], row["Waste Type"], weight=row["Waste Quantity (kg)"])

    # Area to Collection Frequency (count occurrence)
    G.add_edge(row["Area Name"], row["Collection Frequency"], weight=1)

# Store the graph in GraphML format for analysis
graph_filename = "waste_management_graph.graphml"
nx.write_graphml(G, graph_filename)

graph_filename
