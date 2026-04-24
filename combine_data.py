import pandas as pd

# Load all files
X_train = pd.read_csv("X_train.csv")
X_test = pd.read_csv("X_test.csv")
y_train = pd.read_csv("y_train.csv")
y_test = pd.read_csv("y_test.csv")

# Combine train + test
X = pd.concat([X_train, X_test], axis=0).reset_index(drop=True)
y = pd.concat([y_train, y_test], axis=0).reset_index(drop=True)

# Rename target column
y.columns = ["target"]
# Combine features + target
final_df = pd.concat([X, y], axis=1)

# Save smaller sample to avoid storage issue
final_df = final_df.sample(n=min(50000, len(final_df)), random_state=42)

final_df.to_csv("final_dataset_sample.csv", index=False)

print("✅ final_dataset_sample.csv created successfully!")