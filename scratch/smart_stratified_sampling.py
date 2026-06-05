import os
import pandas as pd
import numpy as np

def smart_stratified_sampling(csv_path="data/CheXpert/train.csv", output_path="data/CheXpert/train_stratified_subset.csv", policy="u-ones", subset_ratio=0.40):
    if not os.path.exists(csv_path):
        print(f"Error: CSV file not found at {csv_path}")
        return

    # Load and filter for Frontal images
    df = pd.read_csv(csv_path)
    if 'Frontal/Lateral' in df.columns:
        df = df[df['Frontal/Lateral'] == 'Frontal'].reset_index(drop=True)
    
    total_len = len(df)
    target_size = int(total_len * subset_ratio)
    print(f"Original Frontal dataset size: {total_len}")
    print(f"Target subset size ({int(subset_ratio*100)}%): {target_size}")

    # Define the 9 concepts
    concepts = [
        'No Finding', 'Enlarged Cardiomediastinum', 'Lung Opacity', 
        'Lung Lesion', 'Pneumonia', 'Pneumothorax', 'Pleural Other', 
        'Fracture', 'Support Devices'
    ]

    # Preprocess labels based on policy
    df_clean = df.copy()
    for col in concepts:
        df_clean[col] = pd.to_numeric(df_clean[col], errors='coerce').fillna(0.0)
        if policy == "u-ones":
            df_clean[col] = df_clean[col].replace(-1.0, 1.0)
        elif policy == "u-zeros":
            df_clean[col] = df_clean[col].replace(-1.0, 0.0)

    # Compute global positive prevalence rates to identify rare concepts (< 5% prevalence)
    prevalences = {col: (df_clean[col] == 1.0).mean() for col in concepts}
    print("\n--- Original Concept Prevalences ---")
    for col, prev in prevalences.items():
        print(f"{col}: {prev:.4f} ({int(prev * total_len)} positive cases)")

    # Identify rare concepts dynamically (< 5% prevalence) or fallback to predefined list
    rare_concepts = [col for col, prev in prevalences.items() if prev < 0.05 and col != 'No Finding']
    if not rare_concepts:
        # Predefined fallback list if no concept is below 5% under the policy
        rare_concepts = ['Pneumothorax', 'Fracture', 'Pleural Other']
    
    print(f"\nIdentified Rare Concepts: {rare_concepts}")

    # Phase 1: Keep 100% of rows containing at least one rare concept
    phase1_mask = (df_clean[rare_concepts] == 1.0).any(axis=1)
    df_phase1 = df_clean[phase1_mask]
    print(f"Phase 1 (Rare Concepts - 100% Keep) size: {len(df_phase1)}")

    # Check if Phase 1 alone exceeds target size (extremely unlikely for rare concepts)
    if len(df_phase1) >= target_size:
        print("Warning: Phase 1 size exceeds or equals target size. Truncating to target size.")
        df_final = df_phase1.sample(n=target_size, random_state=42).reset_index(drop=True)
    else:
        # Phase 2: Completely normal cases ('No Finding' == 1.0) making up exactly 11% of the target subset size
        target_normal_size = int(target_size * 0.11)
        
        # Exclude already selected Phase 1 rows
        remaining_after_p1_mask = ~phase1_mask
        df_remaining = df_clean[remaining_after_p1_mask]
        
        normal_mask = df_remaining['No Finding'] == 1.0
        df_normal_pool = df_remaining[normal_mask]
        
        print(f"Available normal cases pool: {len(df_normal_pool)} (Target normal size: {target_normal_size})")
        
        if len(df_normal_pool) >= target_normal_size:
            df_phase2 = df_normal_pool.sample(n=target_normal_size, random_state=42)
        else:
            print("Warning: Not enough normal cases. Keeping all available normal cases.")
            df_phase2 = df_normal_pool

        print(f"Phase 2 (Normal Cases - 11% of target) size: {len(df_phase2)}")

        # Phase 3: Common concepts/remaining rows to fill up the target size
        # Exclude selected Phase 1 and Phase 2 indices
        selected_indices = set(df_phase1.index).union(set(df_phase2.index))
        remaining_pool_mask = ~df_clean.index.isin(selected_indices)
        df_remaining_pool = df_clean[remaining_pool_mask]

        remaining_needed = target_size - len(df_phase1) - len(df_phase2)
        print(f"Phase 3 (Remaining Common Concepts) needed: {remaining_needed} (Pool size: {len(df_remaining_pool)})")

        if len(df_remaining_pool) >= remaining_needed:
            df_phase3 = df_remaining_pool.sample(n=remaining_needed, random_state=42)
        else:
            print("Warning: Not enough remaining samples. Keeping all remaining pool.")
            df_phase3 = df_remaining_pool

        # Combine, shuffle and finalize
        df_final_clean = pd.concat([df_phase1, df_phase2, df_phase3], axis=0)
        df_final_clean = df_final_clean.sample(frac=1.0, random_state=42)

        # Retrieve the original un-preprocessed rows matching the selected indices for output
        df_final = df.loc[df_final_clean.index].copy().reset_index(drop=True)
        
    print(f"\nFinal Subset Size: {len(df_final)}")
    
    # Print the final class distribution of the 9 concepts
    print("\n--- Final Subset Concept Prevalences ---")
    for col in concepts:
        # Clean labels temporarily for printing prevalences using the same policy
        col_clean = pd.to_numeric(df_final[col], errors='coerce').fillna(0.0)
        if policy == "u-ones":
            col_clean = col_clean.replace(-1.0, 1.0)
        elif policy == "u-zeros":
            col_clean = col_clean.replace(-1.0, 0.0)
        
        prev = (col_clean == 1.0).mean()
        print(f"{col}: {prev:.4f} ({int(prev * len(df_final))} positive cases)")

    # Save the selected indices/subset
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    df_final.to_csv(output_path, index=False)
    print(f"\nSaved stratified subset to {output_path}")

if __name__ == "__main__":
    smart_stratified_sampling()
