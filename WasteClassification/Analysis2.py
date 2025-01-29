import pandas as pd
import networkx as nx
import statsmodels.api as sm
from sklearn.cluster import KMeans
from mlxtend.frequent_patterns import apriori, association_rules
from sklearn.neighbors import NearestNeighbors
import matplotlib.pyplot as plt

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
    G.add_edge(row["Waste Type"], row["Source"], weight=row["Waste Quantity (kg)"])
    G.add_edge(row["Waste Type"], row["Disposal Method"], weight=row["Waste Quantity (kg)"])
    G.add_edge(row["Area Name"], row["Waste Type"], weight=row["Waste Quantity (kg)"])
    G.add_edge(row["Area Name"], row["Collection Frequency"], weight=1)

# PageRank Centrality
pagerank_centrality = nx.pagerank(G, weight="weight")
pagerank_centrality_df = pd.DataFrame(list(pagerank_centrality.items()), columns=["Node", "PageRank Centrality"])
print("\n✅ **PageRank Centrality:**")
print(pagerank_centrality_df)

# Eccentricity of nodes
eccentricity = nx.eccentricity(G)
eccentricity_df = pd.DataFrame(list(eccentricity.items()), columns=["Node", "Eccentricity"])
print("\n✅ **Eccentricity of nodes:**")
print(eccentricity_df)

# Dominating Set (approximation)
dominating_set = nx.approximation.min_weighted_dominating_set(G)
print("\n✅ **Dominating Set:**")
print(dominating_set)

# Flow analysis (edges with highest weights)
sorted_edges_flow = sorted(G.edges(data=True), key=lambda x: x[2]["weight"], reverse=True)
print("\n✅ **Top Edges by Waste Flow:**")
for u, v, data in sorted_edges_flow[:10]:
    print(f"From {u} to {v} with Waste Flow (kg): {data['weight']}")

# Time-series forecasting for waste quantity prediction per area
df['Date of Collection'] = pd.to_datetime(df['Date of Collection'])
df_area_waste = df.groupby(['Area Name', 'Date of Collection'])['Waste Quantity (kg)'].sum().reset_index()


# Recommend improvements based on recyclability score
high_recyclability_areas = df[df['Recyclability Score (%)'] > 80]
print("\n✅ **Areas with High Recyclability Potential:**")
print(high_recyclability_areas[['Area Name', 'Recyclability Score (%)']])

# Recommendation for optimized collection frequency based on waste quantity
optimized_freq = df.groupby('Area Name')['Collection Frequency'].value_counts().idxmax()
print("\n✅ **Optimized Collection Frequency Recommendations:**")
print(optimized_freq)

# Clustering zones based on waste types using KMeans
df_waste_types = pd.get_dummies(df['Waste Type'])
kmeans = KMeans(n_clusters=3)  # Example: clustering into 3 clusters
df['Cluster'] = kmeans.fit_predict(df_waste_types)

print("\n✅ **Clusters of Zones Based on Waste Types:**")
print(df[['Area Name', 'Cluster']].groupby('Cluster').agg(list))

import pandas as pd
import numpy as np
from mlxtend.frequent_patterns import apriori, association_rules
from sklearn.neighbors import NearestNeighbors
import matplotlib.pyplot as plt
import networkx as nx
from sklearn.preprocessing import LabelEncoder

# Assuming df is your dataframe and it has necessary columns like 'Waste Type', 'Disposal Method', 'Source', 'Collection Frequency'

# Step 1: Association Rule Mining for Waste Management Patterns
df_encoded = pd.get_dummies(df[['Waste Type', 'Disposal Method', 'Source', 'Collection Frequency']])

# Apply Apriori Algorithm to detect frequent itemsets
frequent_itemsets = apriori(df_encoded, min_support=0.05, use_colnames=True)

# Generate association rules with a minimum lift of 1.5
rules = association_rules(frequent_itemsets, metric="lift", min_threshold=1.5)

# Output association rules
print("\n✅ **Association Rules for Waste Management:**")
if not rules.empty:
    print(rules[['antecedents', 'consequents', 'support', 'confidence', 'lift']])
else:
    print("No significant rules found based on the provided dataset.")

# Step 2: Collaborative Filtering-Based Recommendations (Content-Based)
# Convert categorical 'Collection Frequency' to numeric using Label Encoding
label_encoder = LabelEncoder()
df['Collection Frequency'] = label_encoder.fit_transform(df['Collection Frequency'])

# Select relevant numeric features for collaborative filtering
X = df[['Recyclability Score (%)', 'Waste Quantity (kg)', 'Collection Frequency']]

# Handle missing data, if any, by specifying numeric_only=True
X[['Recyclability Score (%)', 'Waste Quantity (kg)', 'Collection Frequency']].fillna(X.mean(numeric_only=True))


# Initialize Nearest Neighbors model with cosine similarity metric
model_nn = NearestNeighbors(n_neighbors=3, metric='cosine')
model_nn.fit(X)

# Step 2.1: Ensure the specific area exists before recommending
area_name = "Zone A"
area_to_recommend = df[df['Area Name'] == area_name]

if area_to_recommend.empty:
    print(f"🚫 **Error:** The area '{area_name}' was not found in the dataset.")
else:
    # Get the nearest neighbors for the selected area
    distances, indices = model_nn.kneighbors(area_to_recommend[['Recyclability Score (%)', 'Waste Quantity (kg)', 'Collection Frequency']])

    # Output the recommended areas
    print(f"\n✅ **Recommended Areas for Similar Waste Management Strategies for {area_name}:**")
    for idx in indices[0]:
        print(df.iloc[idx]['Area Name'])

# Step 3: Visualization of Waste Management Relationships using Graph
# Create a Graph from the waste management data (e.g., Waste Type and Disposal Method connections)
G = nx.Graph()

# Assuming there are relationships between Waste Type and Disposal Method (or other entities)
for _, row in df.iterrows():
    G.add_edge(row['Waste Type'], row['Disposal Method'])

# Visualize the graph with a spring layout
plt.figure(figsize=(12, 12))
pos = nx.spring_layout(G, seed=42)  # Set seed for reproducibility
nx.draw(G, pos, with_labels=True, node_size=3000, node_color='lightblue', font_size=10, font_weight='bold')
plt.title("Waste Management Graph", fontsize=16)
plt.show()
