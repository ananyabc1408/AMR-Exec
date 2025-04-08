import pandas as pd
import numpy as np
from faker import Faker
import random

# Initialize Faker
fake = Faker()

# Define the number of samples (rows)
num_samples = 5000

# Define the data
data = {
    "pathogen": [random.choice(["Salmonella", "E. coli", "Campylobacter"]) for _ in range(num_samples)],
    "antibiotic": [random.choice(["Amoxicillin", "Ciprofloxacin", "Erythromycin", "Azithromycin", "Trimethoprim",
                                  "Doxycycline", "Ceftriaxone", "Gentamicin", "Norfloxacin", "Chloramphenicol",
                                  "Nitrofurantoin", "Levofloxacin"]) for _ in range(num_samples)],
    "resistance_level": [random.choice(["High", "Medium", "Low"]) for _ in range(num_samples)],
    "collection_date": [fake.date_this_decade() for _ in range(num_samples)],
    "location": [fake.city() for _ in range(num_samples)]
}

# Create a DataFrame
df = pd.DataFrame(data)
# Save the DataFrame to a CSV file
df.to_csv("antimicrobial_resistance_data_large.csv", index=False)

print("CSV file created successfully!")
