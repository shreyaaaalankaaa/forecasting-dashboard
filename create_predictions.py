import numpy as np
import pandas as pd
# Load y_test
y_true = pd.read_csv("y_test.csv")

# Create fake predictions (for now)
y_pred = y_true.copy()
y_pred["y_pred"] = y_true.iloc[:, 0] + np.random.normal(0, 0.1, len(y_true))

# Final dataframe
final_df = pd.DataFrame({
    "y_true": y_true.iloc[:, 0],
    "y_pred": y_pred["y_pred"]
})

# Save file
final_df.to_csv("predictions.csv", index=False)

print("predictions.csv created successfully!")