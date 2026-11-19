
import pandas as pd

# Load the original project dataset
df = pd.read_csv("data/ML_Dataset.csv")

print("Available columns:", df.columns.tolist())

# Find the electricity demand column
matches = [
    col for col in df.columns
    if "power" in col.lower() and "demand" in col.lower()
]

if not matches:
    raise ValueError(
        "Could not identify the demand column. "
        "Check the column names printed above."
    )

# Extract the latest 289 valid readings
demand = pd.to_numeric(df[matches[0]], errors="coerce").dropna().tail(289)

if len(demand) != 289:
    raise ValueError("Dataset does not contain 289 valid readings.")

# Save a CSV accepted by Manual Mode
pd.DataFrame({"Power demand": demand}).to_csv(
    "demo_demand.csv", index=False
)

print("Created demo_demand.csv with exactly 289 readings.")
