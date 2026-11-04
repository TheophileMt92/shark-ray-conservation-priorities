import sys
import os
import numpy as np
import pandas as pd
import pyreadr
import matplotlib.pyplot as plt

# Directory path containing the .npz files for IUCN
dir_path = 'CAPTAIN_2_outputs/iucn_pred_20250720'  # folder with the CAPTAIN 2 prediction files (.npz)

# Function to process a single .npz file
def process_file(file_path):
    # Load the .npz file
    data = np.load(file_path)
    
    # Extract the protection_matrix
    if 'protection_matrix' in data:
        protection_matrix = data['protection_matrix']
        
        # Flatten the matrix to get a 1D array of all grid cell values
        # Each element represents the priority for a single grid cell
        flattened_priorities = protection_matrix.flatten()
        
        # Create a DataFrame with grid cell IDs and priority values
        df = pd.DataFrame({
            'PUID': np.arange(1, len(flattened_priorities) + 1),  # Generate IDs from 1 to n
            'Priority': flattened_priorities
        })
        
        return df
    else:
        print(f"No protection_matrix found in {file_path}")
        return None

# Get all .npz files in the directory
npz_files = [f for f in os.listdir(dir_path) if f.endswith('.npz')]

# Process all files
results = []
for file in npz_files:
    file_path = os.path.join(dir_path, file)
    try:
        result = process_file(file_path)
        if result is not None:
            results.append(result)
            print(f"Processed: {file}")
    except Exception as e:
        print(f"Error processing file {file_path}: {e}")

# If we have results to combine
if results:
    # Combine all results, averaging the Priority
    # First ensure all DataFrames have the same structure
    combined_results = pd.DataFrame({'PUID': results[0]['PUID'].values})
    combined_results['Priority'] = 0
    
    for result in results:
        # Make sure PUIDs align
        if not np.array_equal(combined_results['PUID'].values, result['PUID'].values):
            print("Warning: PUID mismatch between results. Realigning...")
            result = result.set_index('PUID').reindex(combined_results['PUID']).reset_index()
        
        combined_results['Priority'] += result['Priority']
    
    combined_results['Priority'] /= len(results)  # Calculate the average
    
    # Generate summary statistics
    summary = combined_results.describe()
    
    print("\nSummary of averaged results:")
    print(summary)
    
    # Show unique values of Priority
    unique_priorities = combined_results['Priority'].unique()
    print(f"\nNumber of unique Priority values: {len(unique_priorities)}")
    print("First 20 unique priority values:")
    print(np.sort(unique_priorities)[:20])  # Show just the first 20 for readability
    
    # Calculate some additional statistics
    nonzero_priorities = combined_results[combined_results['Priority'] > 0]['Priority']
    print(f"\nNon-zero priority cells: {len(nonzero_priorities)} ({len(nonzero_priorities)/len(combined_results)*100:.2f}%)")
    
    # Save as RDS with IUCN in the name
    budget = 0.1  # Update with your budget value
    replicates = len(results)
    rds_path = f'CAPTAIN2_IUCN_full_results_averaged_budget{budget}_replicates{replicates}.rds'
    pyreadr.write_rds(rds_path, combined_results)
    print(f"Saved combined results to {rds_path}")
    
    # Save the shape information for reference
    with open('grid_shape_info_IUCN.txt', 'w') as f:
        f.write(f"Original grid shape: 323 rows x 720 columns\n")
        f.write(f"Total grid cells: {323 * 720}\n")
        f.write(f"Flattening order: row-major (C-style)\n")
    print("Saved grid shape information to grid_shape_info_IUCN.txt")
    
    # Visualize the priority values from the first file as a 2D grid
    if npz_files:
        example_file = os.path.join(dir_path, npz_files[0])
        data = np.load(example_file)
        if 'protection_matrix' in data:
            protection_matrix = data['protection_matrix']
            
            # Create heatmap of single replicate
            plt.figure(figsize=(12, 6))
            plt.imshow(protection_matrix, cmap='viridis')
            plt.colorbar(label='Priority Value')
            plt.title('IUCN Conservation Priority Grid (First Replicate)')
            plt.xlabel('Longitude (Column Index)')
            plt.ylabel('Latitude (Row Index)')
            plt.savefig('IUCN_priority_grid_single_replicate.png')
            print("Saved IUCN priority grid visualization (single replicate) to IUCN_priority_grid_single_replicate.png")
            
            # Create heatmap of averaged results
            avg_protection_matrix = combined_results['Priority'].values.reshape(323, 720)
            plt.figure(figsize=(12, 6))
            plt.imshow(avg_protection_matrix, cmap='viridis')
            plt.colorbar(label='Priority Value')
            plt.title('IUCN Conservation Priority Grid (Averaged Across Replicates)')
            plt.xlabel('Longitude (Column Index)')
            plt.ylabel('Latitude (Row Index)')
            plt.savefig('IUCN_priority_grid_averaged.png')
            print("Saved IUCN priority grid visualization (averaged) to IUCN_priority_grid_averaged.png")
else:
    print("No valid results to process.")