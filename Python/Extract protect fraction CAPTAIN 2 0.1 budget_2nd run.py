import numpy as np
import os
import pandas as pd
import pyreadr
import glob

# Directories containing the CAPTAIN 2 prediction files (.npz) for each index
edge_dir = 'CAPTAIN_2_outputs/edge_pred_20250720'
fuse_dir = 'CAPTAIN_2_outputs/fuse_pred_20250720'
iucn_dir = 'CAPTAIN_2_outputs/iucn_pred_20250720'

# Directory containing the input TIF files with species names
input_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', 'R', 'Data', 'tif files continental')

# Function to extract protected_range_fraction from all files in a directory
def extract_protected_fractions(dir_path):
    npz_files = [f for f in os.listdir(dir_path) if f.endswith('.npz')]
    all_fractions = []
    
    for file in npz_files:
        file_path = os.path.join(dir_path, file)
        try:
            data = np.load(file_path)
            if 'protected_range_fraction' in data:
                fractions = data['protected_range_fraction']
                # Add to our collection
                all_fractions.append(fractions)
                print(f"Extracted fractions from: {file}")
            else:
                print(f"No protected_range_fraction in {file}")
        except Exception as e:
            print(f"Error processing file {file_path}: {e}")
    
    # Stack all arrays and calculate the average fraction for each species
    if all_fractions:
        stacked_fractions = np.stack(all_fractions)
        avg_fractions = np.mean(stacked_fractions, axis=0)
        return avg_fractions
    else:
        print(f"No valid fractions found in {dir_path}")
        return None

# Get species names from TIF files
def get_species_from_tifs(input_dir):
    # Get all TIF files
    tif_files = glob.glob(os.path.join(input_dir, "*.tif"))
    
    # Extract species names from filenames
    species_names = []
    for tif_file in tif_files:
        # Extract filename without extension and path
        basename = os.path.basename(tif_file)
        species_name = os.path.splitext(basename)[0]
        species_names.append(species_name)
    
    # Sort alphabetically to match expected order
    species_names.sort()
    
    print(f"Found {len(species_names)} species names from TIF files")
    print(f"First 5 species: {species_names[:5]}")
    print(f"Last 5 species: {species_names[-5:]}")
    
    return species_names

# Get the species names from TIF files
species_names = get_species_from_tifs(input_dir)

# Extract fractions for each index
print("\nExtracting EDGE2 protected fractions...")
edge_fractions = extract_protected_fractions(edge_dir)

print("\nExtracting FUSE protected fractions...")
fuse_fractions = extract_protected_fractions(fuse_dir)

print("\nExtracting IUCN protected fractions...")
iucn_fractions = extract_protected_fractions(iucn_dir)

# Create a DataFrame with all the fractions
print("\nCreating DataFrame...")
species_count = len(edge_fractions) if edge_fractions is not None else 0

# Check if the count of species matches
if len(species_names) != species_count:
    print(f"WARNING: Number of species from TIF files ({len(species_names)}) doesn't match the number of species in the CAPTAIN2 outputs ({species_count})")
    # Use generic species identifiers if counts don't match
    species_column = [f"Species_{i+1}" for i in range(species_count)]
else:
    print(f"Number of species matches: {species_count}")
    species_column = species_names

# Initialize the DataFrame
df = pd.DataFrame({'Species': species_column})

# Add each index's fractions if available
if edge_fractions is not None:
    df['EDGE2'] = edge_fractions
if fuse_fractions is not None:
    df['FUSE'] = fuse_fractions
if iucn_fractions is not None:
    df['IUCN'] = iucn_fractions

# Print summary statistics
print("\nDataFrame Summary:")
print(f"Total species: {len(df)}")
print(f"Columns: {df.columns.tolist()}")
print("\nSample data (first 5 rows):")
print(df.head())
print("\nSummary statistics:")
print(df.describe())

# Save as RDS file
rds_filename = 'CAPTAIN2_protected_range_fractions_2ndrun.rds'
pyreadr.write_rds(rds_filename, df)
print(f"\nSaved to {rds_filename}")

# Also save as CSV for easier inspection
csv_filename = 'CAPTAIN2_protected_range_fractions_2ndrun.csv'
df.to_csv(csv_filename, index=False)
print(f"Also saved to {csv_filename}")