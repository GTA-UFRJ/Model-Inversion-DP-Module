import pandas as pd

# Load the CSV
df = pd.read_csv("new_results_with_seed.csv")

# Define seeds to remove
seeds_to_remove = [18, 20, 26, 28, 42, 45, 46, 47, 50, 51]

# Remove rows where Seed is in the list
df = df[~df["Seed"].isin(seeds_to_remove)]

# Save the updated CSV
df.to_csv("updated_new_results_with_seed.csv", index=False)
