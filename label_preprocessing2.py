import pandas as pd
import re
from src.constants import entity_unit_map  # Import the correct mapping

def preprocess_labels(df):
    def extract_value_and_unit(entity_value):
        match = re.match(r'(\d+(?:\.\d+)?)\s*(\w+)', str(entity_value))
        if match:
            value, unit = match.groups()
            return float(value), unit.lower()
        return None, None

    def normalize_unit(unit, entity_name):
        allowed_units = entity_unit_map.get(entity_name, set())  # Get allowed units for the entity
        if unit in allowed_units:
            return unit
        # Instead of returning None, return a default value or a flag
        return "invalid"  # or return a default unit like "unknown"

    # Extract value and unit
    df['value'], df['unit'] = zip(*df['entity_value'].map(extract_value_and_unit))

    # Log the number of rows before normalization
    print(f"Rows before normalization: {len(df)}")
    
    # Normalize units
    df['normalized_unit'] = df.apply(lambda row: normalize_unit(row['unit'], row['entity_name']), axis=1)

    # Log the number of rows after normalization
    print(f"Rows after normalization: {len(df)}")
    
    # (Optional) You can set a flag for invalid rows instead of dropping
    # df['is_valid'] = df['normalized_unit'] != "invalid"
    
    return df

# Load the training data
train_df = pd.read_csv('dataset/train.csv')

# Sample 100 rows from the training data (similar to the image sampling logic)
sampled_train_df = train_df.sample(n=100, random_state=42).reset_index(drop=True)

# Preprocess the labels of the sampled data
preprocessed_train_df = preprocess_labels(sampled_train_df)

# Save the preprocessed data
preprocessed_train_df.to_csv('preprocessed/train_labels.csv', index=False)

print("Label preprocessing complete!")
