import pandas as pd
import numpy as np
from sklearn.cluster import KMeans

# Load player data from a CSV file
player_data = pd.read_csv("player_data.csv")

# Define the features to use for clustering
features = ["height", "weight", "sprinting_speed", "shooting_speed"]

# Normalize the data to have mean = 0 and variance = 1
normalized_data = (player_data[features] - player_data[features].mean()) / player_data[features].std()

# Apply k-means clustering with k=3 
kmeans = KMeans(n_clusters=3, random_state=0).fit(normalized_data)

# Get the cluster labels for each player
cluster_labels = kmeans.labels_

# Add the cluster labels to the player data
player_data["cluster"] = cluster_labels

# Define the player positions and the optimal number of players per position
positions = {"attacker": 4, "midfielder": 3, "defender": 4, "goalkeeper": 1}

# Group the players by cluster
cluster_groups = player_data.groupby("cluster")

# Place the players in their optimal positions
for position, num_players in positions.items():
    position_players = pd.DataFrame()
    for _, group in cluster_groups:
        # Sort the players in the cluster by the feature that is most important for the position
        sorted_group = group.sort_values(by=features.index(position.split("_")[0]), ascending=False)
        # Take the top num_players players from the sorted group
        position_players = pd.concat([position_players, sorted_group.head(num_players)])
    # Print the players in the position
    print(f"{position.capitalize()}s:")
    print(position_players[["player_name", "cluster"]])