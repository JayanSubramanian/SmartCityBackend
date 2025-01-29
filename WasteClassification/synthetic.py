import pandas as pd
import random
from datetime import datetime, timedelta

# Define waste categories
waste_categories = [
    "Organic Waste", "Plastic Waste", "Metal Waste", "Glass Waste", "E-Waste",
    "Hazardous Waste", "Paper Waste", "Textile Waste", "Construction & Demolition Waste",
    "Mixed Waste", "Biomedical Waste", "Agricultural Waste"
]

# Define sources of waste
sources = ["Household", "Industrial", "Commercial", "Hospital", "Agricultural"]

# Define disposal methods
disposal_methods = ["Landfill", "Recycling", "Incineration", "Composting"]

# Define areas
areas = ["Zone A", "Zone B", "Zone C", "Zone D", "Zone E"]

# Generate random data
num_records = 500  # Number of records to generate
data = []

start_date = datetime(2023, 1, 1)
for _ in range(num_records):
    record = {
        "Area Name": random.choice(areas),
        "Date of Collection": (start_date + timedelta(days=random.randint(0, 365))).strftime('%Y-%m-%d'),
        "Waste Type": random.choice(waste_categories),
        "Waste Quantity (kg)": round(random.uniform(5, 500), 2),
        "Source": random.choice(sources),
        "Recyclability Score (%)": round(random.uniform(10, 90), 2),
        "Disposal Method": random.choice(disposal_methods),
        "Collection Frequency": random.choice(["Daily", "Weekly", "Bi-Weekly", "Monthly"])
    }
    data.append(record)

# Convert to DataFrame
df = pd.DataFrame(data)

# Save to CSV file
csv_filename = "waste_classification_data.csv"
df.to_csv(csv_filename, index=False)

# Display first few rows
df.head()
