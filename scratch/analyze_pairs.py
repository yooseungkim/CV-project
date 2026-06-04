import pandas as pd
import os
import re

csv_path = 'data/derm7pt/meta/meta.csv'
df = pd.read_csv(csv_path)

print("Total rows:", len(df))

def clean_filename(path):
    if pd.isna(path): return ""
    return os.path.basename(path).lower()

df['clinic_file'] = df['clinic'].apply(clean_filename)
df['derm_file'] = df['derm'].apply(clean_filename)

# 1. Check rows where clinic contains 'bis' or derm contains 'bis'
bis_rows = df[df['clinic_file'].str.contains('bis') | df['derm_file'].str.contains('bis')]
print(f"Number of rows containing 'bis' in clinic/derm filenames: {len(bis_rows)}")

# Print a few examples of bis rows
print("\nExamples of rows with 'bis' filenames:")
print(bis_rows[['case_num', 'clinic', 'derm', 'diagnosis']].head(10).to_string())

# 2. Check if there are matching non-bis rows for these bis rows.
# To do this, we extract the base number/id from the filename (e.g., 'aal099bis.jpg' -> 'aal099')
# and see if there is another row in the CSV that has 'aal099.jpg' or similar.
def extract_base(filename):
    if not filename: return ""
    name = filename.split('.')[0]
    # Remove 'bis' or other suffixes
    name = re.sub(r'bis$', '', name)
    return name

df['clinic_base'] = df['clinic_file'].apply(extract_base)
df['derm_base'] = df['derm_file'].apply(extract_base)

# Let's see if the same clinic_base appears in multiple rows
clinic_base_counts = df['clinic_base'].value_counts()
multiple_clinic = clinic_base_counts[clinic_base_counts > 1]
print(f"\nNumber of clinic bases appearing in multiple rows: {len(multiple_clinic)}")
if len(multiple_clinic) > 0:
    print("Examples:")
    print(multiple_clinic.head(10))
    
    # Show the full rows for one of them
    example_base = multiple_clinic.index[0]
    print(f"\nRows matching clinic_base '{example_base}':")
    print(df[df['clinic_base'] == example_base][['case_num', 'clinic', 'derm', 'diagnosis', 'clinic_file', 'derm_file']].to_string())

# Let's see if the same derm_base appears in multiple rows
derm_base_counts = df['derm_base'].value_counts()
multiple_derm = derm_base_counts[derm_base_counts > 1]
print(f"\nNumber of derm bases appearing in multiple rows: {len(multiple_derm)}")
if len(multiple_derm) > 0:
    print("Examples:")
    print(multiple_derm.head(10))

# 3. How are they split in train/valid/test_indexes.csv?
# Let's load the index files and see if any clinic_base or derm_base spans across splits.
meta_dir = 'data/derm7pt/meta'
train_idx = pd.read_csv(os.path.join(meta_dir, 'train_indexes.csv'))['indexes'].tolist()
valid_idx = pd.read_csv(os.path.join(meta_dir, 'valid_indexes.csv'))['indexes'].tolist()
test_idx = pd.read_csv(os.path.join(meta_dir, 'test_indexes.csv'))['indexes'].tolist()

split_map = {}
for idx in train_idx: split_map[idx] = 'train'
for idx in valid_idx: split_map[idx] = 'valid'
for idx in test_idx: split_map[idx] = 'test'

df['split'] = df.index.map(split_map)

# Check if same base is in different splits (which would be data leakage!)
df['base_id'] = df['clinic_base'].str.cat(df['derm_base'], sep='_')
base_split_groups = df.groupby('clinic_base')['split'].nunique()
leaked_bases = base_split_groups[base_split_groups > 1]
print(f"\nNumber of clinic bases split across train/val/test (data leakage!): {len(leaked_bases)}")
if len(leaked_bases) > 0:
    print(df[df['clinic_base'].isin(leaked_bases.index)][['case_num', 'clinic_base', 'split', 'diagnosis']])
