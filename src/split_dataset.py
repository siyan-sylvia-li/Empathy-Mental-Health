import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split

# Set random seed for reproducibility
np.random.seed(42)

# Read the full dataset
df = pd.read_csv('dataset/emotional-reactions-reddit.csv')

# First split: 80% train, 20% temp (which will be split into val and test)
train_df, temp_df = train_test_split(df, test_size=0.2, random_state=42)

# Second split: 50% of temp for validation, 50% for test (which gives 10% each of the original data)
val_df, test_df = train_test_split(temp_df, test_size=0.5, random_state=42)

# Save the splits
train_df.to_csv('dataset/train_er.csv', index=False)
val_df.to_csv('dataset/val_er.csv', index=False)
test_df.to_csv('dataset/test_er.csv', index=False)

# Print the sizes of each split
print(f"Total dataset size: {len(df)}")
print(f"Training set size: {len(train_df)} ({len(train_df)/len(df)*100:.1f}%)")
print(f"Validation set size: {len(val_df)} ({len(val_df)/len(df)*100:.1f}%)")
print(f"Test set size: {len(test_df)} ({len(test_df)/len(df)*100:.1f}%)") 


# Read the full dataset
df = pd.read_csv('dataset/interpretations-reddit.csv')

# First split: 80% train, 20% temp (which will be split into val and test)
train_df, temp_df = train_test_split(df, test_size=0.2, random_state=42)

# Second split: 50% of temp for validation, 50% for test (which gives 10% each of the original data)
val_df, test_df = train_test_split(temp_df, test_size=0.5, random_state=42)

# Save the splits
train_df.to_csv('dataset/train_int.csv', index=False)
val_df.to_csv('dataset/val_int.csv', index=False)
test_df.to_csv('dataset/test_int.csv', index=False)

# Print the sizes of each split
print(f"Total dataset size: {len(df)}")
print(f"Training set size: {len(train_df)} ({len(train_df)/len(df)*100:.1f}%)")
print(f"Validation set size: {len(val_df)} ({len(val_df)/len(df)*100:.1f}%)")
print(f"Test set size: {len(test_df)} ({len(test_df)/len(df)*100:.1f}%)") 


# Read the full dataset
df = pd.read_csv('dataset/explorations-reddit.csv')

# First split: 80% train, 20% temp (which will be split into val and test)
train_df, temp_df = train_test_split(df, test_size=0.2, random_state=42)

# Second split: 50% of temp for validation, 50% for test (which gives 10% each of the original data)
val_df, test_df = train_test_split(temp_df, test_size=0.5, random_state=42)

# Save the splits
train_df.to_csv('dataset/train_exp.csv', index=False)
val_df.to_csv('dataset/val_exp.csv', index=False)
test_df.to_csv('dataset/test_exp.csv', index=False)

# Print the sizes of each split
print(f"Total dataset size: {len(df)}")
print(f"Training set size: {len(train_df)} ({len(train_df)/len(df)*100:.1f}%)")
print(f"Validation set size: {len(val_df)} ({len(val_df)/len(df)*100:.1f}%)")
print(f"Test set size: {len(test_df)} ({len(test_df)/len(df)*100:.1f}%)") 