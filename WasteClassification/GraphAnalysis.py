import networkx as nx
import community as community_louvain  # Correct import
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd

# -------------------- LOAD GRAPHML FILE -------------------- #
graphml_file = "waste_management_graph.graphml"  # Replace with your file path
G = nx.read_graphml(graphml_file)

# -------------------- COMMUNITY DETECTION -------------------- #
partition = community_louvain.best_partition(G)  # Louvain clustering

# Assign colors to communities
unique_communities = set(partition.values())
color_map = {comm: sns.color_palette("Set1", len(unique_communities))[i] for i, comm in enumerate(unique_communities)}
node_colors = [color_map[partition[node]] for node in G.nodes()]

# -------------------- VISUALIZE GRAPH -------------------- #
plt.figure(figsize=(12, 10))
pos = nx.spring_layout(G, seed=42)
nx.draw(G, pos, node_color=node_colors, with_labels=True, node_size=600, edge_color='gray', font_size=8)
plt.title("Waste Management Graph with Community Detection")
plt.show()


# -------------------- NETWORK ANALYSIS -------------------- #
# Degree Centrality (importance of nodes based on connections)
degree_centrality = nx.degree_centrality(G)
top_degree_nodes = sorted(degree_centrality.items(), key=lambda x: x[1], reverse=True)[:5]

# Betweenness Centrality (nodes acting as bridges)
betweenness_centrality = nx.betweenness_centrality(G)
top_betweenness_nodes = sorted(betweenness_centrality.items(), key=lambda x: x[1], reverse=True)[:5]

# Density of the network
graph_density = nx.density(G)

# -------------------- PRINT ANALYSIS RESULTS -------------------- #
print(f"\n🔹 **Graph Density:** {graph_density:.4f}")
print(f"\n🔹 **Top 5 Nodes by Degree Centrality:**")
for node, value in top_degree_nodes:
    print(f"   {node}: {value:.4f}")

print(f"\n🔹 **Top 5 Nodes by Betweenness Centrality:**")
for node, value in top_betweenness_nodes:
    print(f"   {node}: {value:.4f}")

