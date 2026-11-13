# Data preparation for the CAPTAIN prioritisations (species grid, range rasters and conservation metrics).
# Copied from the original project folder ("CAPTAIN in R/CAPTAIN sharks/CAPTAIN_data_preparation_high seas_vs_continental waters.R");
# input paths refer to that folder and do not resolve within this repository. Provided for transparency.
# The final step (species table used as CAPTAIN input) is reproduced from repository files in 01_data_preparation.qmd.
# November 2026: the species tables are now built by matching species IDs (build_metrics_table()) instead of cbind.data.frame(),
# which had attached the IUCN, FUSE and EDGE2 values to the wrong species names.
# November 2026: 32 species named differently in sharks_iucn_final are matched through iucn_name_corrections.csv
# instead of being assigned the most common IUCN status (Least Concern).

library(sf)
library(terra)
library(rlang)
library(here)
library(tidyverse)

# First database, species grid: pu_vsp ----
# Loading the data
load(here::here("Data", "puvsp_marine.Rdata"))

# Create global species lookup table
global_species_lookup <- data.frame(
  species_name = names(puvsp_marine)[4:ncol(puvsp_marine)],
  species_id = 1:length(names(puvsp_marine)[4:ncol(puvsp_marine)])
)

High_seas = st_read(here::here("Data", "World_High_Seas_v2_20241010", "High_Seas_v2.shp"))

# Create template raster
template_raster <- rast(ext(High_seas), resolution = 0.5)

# Rasterize high seas polygon
highseas_raster <- rasterize(High_seas, template_raster, field = 1, background = 0)

# Plot to verify
plot(highseas_raster, 
     main = "High Seas (1) vs Continental Waters (0)",
     col = c("lightblue", "darkblue"))

# Convert points to SpatVector for extraction
points_coords <- vect(puvsp_marine, geom = c("lon", "lat"), crs = crs(highseas_raster))

# Extract values
highseas_values <- terra::extract(highseas_raster, points_coords)

# Create two separate datasets
highseas_species <- puvsp_marine[highseas_values[,2] == 1, ]
continental_species <- puvsp_marine[highseas_values[,2] == 0, ]

# Modified function to process the datasets using global lookup
process_dataset <- function(df, species_lookup) {
  # Print original species count
  cat("Original unique species count:", length(unique(names(df)[-(1:3)])), "\n")
  
  df_long <- df %>%
    dplyr::mutate(across(4:ncol(.), ~ifelse(. == 0, NA, .))) %>%
    dplyr::mutate(id = dplyr::row_number()) %>%
    dplyr::select(-c(lon, lat)) %>%
    dplyr::filter(if_any(4:ncol(.), ~!is.na(.))) %>%
    tidyr::pivot_longer(cols = -id,
                        names_to = "species_name",
                        values_to = "amount",
                        values_drop_na = TRUE)
  
  # Print species names before join
  cat("\nSample of species names before join:\n")
  print(head(unique(df_long$species_name)))
  
  # Join with lookup table
  result <- df_long %>%
    dplyr::left_join(species_lookup, by = "species_name") %>%
    dplyr::select(sp = species_id, id, amount)
  
  # Print summary after join
  cat("\nSpecies IDs after join:\n")
  print(summary(result$sp))
  
  return(result)
}

# Apply the transformation to both datasets using the same lookup table
highseas_puvsp <- process_dataset(highseas_species, global_species_lookup)
continental_puvsp <- process_dataset(continental_species, global_species_lookup)

# Save the global species lookup for use in subsequent steps
saveRDS(global_species_lookup, 
        here::here("Data", "My dataframes", "global_species_lookup.rds"))

#puvsp: Species ID, PU ID and Species Presence
#Not saving this dataframe since it needs harmonising with the shark_conservation metrics 

# Prepare coordinates ----

# Function to process coordinates
prepare_coordinates <- function(df, output_name) {
  coords <- df %>%
    dplyr::transmute(
      FID = dplyr::row_number() - 1,
      id = dplyr::row_number(),
      Coord_x = lon,
      Coord_y = lat
    )
  
  # Write the coordinates file (without FID)
  write.csv(coords[,-1], 
            here::here("Data", "My dataframes", paste0(output_name, "_pu_coords.csv")), 
            row.names = FALSE)
  
  return(coords)
}

# Process both datasets
highseas_coordinates <- prepare_coordinates(highseas_species, "highseas")
continental_coordinates <- prepare_coordinates(continental_species, "continental")

# Fishing datasets ---- 
# First create separate fishing predictions dataframes
load("Data/Raw/predicted_fishing_everywhere_05Deg.Rdata")

Fishing_preds <- prediction_data_env %>%
  dplyr::rename(Coord_x = 1, Coord_y = 2)

# Process for high seas
highseas_fishing <- highseas_coordinates %>%
  dplyr::select(id, Coord_x, Coord_y) %>%
  dplyr::left_join(Fishing_preds, by = c("Coord_x", "Coord_y")) %>%
  dplyr::select(id, predicted_fishing_hours) %>%
  dplyr::mutate(
    predicted_fishing_hours = dplyr::coalesce(predicted_fishing_hours, 0.1),
    predicted_fishing_hours = log1p(predicted_fishing_hours)
  )

# Calculate mean for high seas
highseas_mean_fishing <- mean(highseas_fishing$predicted_fishing_hours, na.rm = TRUE)

# Replace with mean and prepare for export
highseas_fishing_final <- highseas_fishing %>%
  dplyr::mutate(predicted_fishing_hours = highseas_mean_fishing) %>%
  dplyr::rename(cost = predicted_fishing_hours)

# Process for continental waters
continental_fishing <- continental_coordinates %>%
  dplyr::select(id, Coord_x, Coord_y) %>%
  dplyr::left_join(Fishing_preds, by = c("Coord_x", "Coord_y")) %>%
  dplyr::select(id, predicted_fishing_hours) %>%
  dplyr::mutate(
    predicted_fishing_hours = dplyr::coalesce(predicted_fishing_hours, 0.1),
    predicted_fishing_hours = log1p(predicted_fishing_hours)
  )

# Calculate mean for continental waters
continental_mean_fishing <- mean(continental_fishing$predicted_fishing_hours, na.rm = TRUE)

# Replace with mean and prepare for export
continental_fishing_final <- continental_fishing %>%
  dplyr::mutate(predicted_fishing_hours = continental_mean_fishing) %>%
  dplyr::rename(cost = predicted_fishing_hours)

# Save the files
write.csv(highseas_fishing_final, 
          here::here("Data", "My dataframes", "highseas_pu_fishing.csv"), 
          row.names = FALSE)

write.csv(continental_fishing_final, 
          here::here("Data", "My dataframes", "continental_pu_fishing.csv"), 
          row.names = FALSE)

# Print summary statistics
cat("High seas mean fishing hours (log1p):", highseas_mean_fishing, "\n")
cat("Continental waters mean fishing hours (log1p):", continental_mean_fishing, "\n")

# Create Species Biodiversity Index Values lists: shark_conservation_metrics.rds ----
# Load global species lookup created in first part
global_species_lookup <- readRDS(here::here("Data", "My dataframes", "global_species_lookup.rds"))

# Read data
shark.info <- readRDS("Data/Raw/CAPTAIN_SharkInformation.rds")
shark.EDGE <- readRDS("Data/Raw/sharks_EDGE_final.rds")
shark.FUSE <- read.csv("Data/Raw/shark_sp_metrics.csv")
shark.IUCN <- read.csv("Data/Raw/species_updated.iucn_no.syn.csv")

# Function to process and join metrics
process_and_join_metric <- function(info_data, metric_data, metric_name) {
  metric_data %>%
    dplyr::select(Species, !!metric_name) %>%
    dplyr::mutate(
      Species = gsub("_", " ", Species),
      !!metric_name := vegan::decostand(!!sym(metric_name), "range")
    ) %>%
    {
      missing_species <- setdiff(info_data$Species, .$Species)
      message(paste("Missing species in", metric_name, ":", length(missing_species)))
      .
    } %>%
    dplyr::right_join(info_data, by = "Species")
}

# Process and join all metrics
shark.info <- shark.info %>%
  process_and_join_metric(shark.EDGE, "ED") %>%
  process_and_join_metric(shark.EDGE, "EDGE_P100") %>%
  process_and_join_metric(shark.FUSE, "FUSE") %>%
  process_and_join_metric(shark.FUSE, "EDGE2") %>%
  process_and_join_metric(shark.FUSE, "FUn") %>%
  process_and_join_metric(shark.FUSE, "FSp")

# Reorder columns
shark.info <- shark.info %>%
  dplyr::relocate(c(ED, EDGE_P100, EDGE2, FUn, FSp, FUSE), .after = IUCN)

# Transforming IUCN status
shark.info <- shark.info %>%
  mutate(IUCN = case_when(
    IUCN %in% c("NE", "DD") ~ NA_real_,
    IUCN == "LC" ~ 0,
    IUCN == "NT" ~ 1,
    IUCN == "VU" ~ 2,
    IUCN == "EN" ~ 3,
    IUCN == "CR" ~ 4,
    TRUE ~ as.numeric(as.character(IUCN))
  )) %>%
  mutate(IUCN = vegan::decostand(IUCN, "range", na.rm = TRUE))

# Filter to species with complete metrics
shark.info_complete <- shark.info %>%
  drop_na(ED, EDGE_P100, FUSE, EDGE2, FUn, FSp)

cat("Species with complete metrics:", nrow(shark.info_complete), "\n")

# Modified filter_puvsp_to_metrics function
filter_puvsp_to_metrics <- function(puvsp_data, shark_info_complete, species_lookup) {
  # Get species columns (assuming columns 4+ are species)
  species_cols <- names(puvsp_data)[4:ncol(puvsp_data)]
  
  # Keep only species that are in shark_info_complete and lookup
  species_to_keep <- species_cols[species_cols %in% shark.info_complete$Species]
  
  # Select these species plus the coordinate columns
  filtered_puvsp <- puvsp_data %>%
    select(id, lon, lat, all_of(species_to_keep))
  
  cat("Original species count:", length(species_cols), "\n")
  cat("Species after filtering:", length(species_to_keep), "\n")
  
  return(filtered_puvsp)
}

process_zone_metrics <- function(shark.info, puvsp_zone, zone_name, species_lookup) {
  species_in_zone <- names(puvsp_zone)[4:ncol(puvsp_zone)]
  
  zone_shark_info <- shark.info %>%
    filter(Species %in% species_in_zone) %>%
    left_join(species_lookup, by = c("Species" = "species_name"))
  
  print(paste("Number of species retained in", zone_name, ":", nrow(zone_shark_info)))
  
  # Add diagnostic check for IUCN values
  cat("\nIUCN value check before processing:\n")
  print(summary(zone_shark_info$IUCN))
  
  process_dataset <- function(shark.info, puvsp, metric) {
    # Drop NAs and extract species IDs
    info <- drop_na(shark.info, all_of(metric))
    species_ids <- info$species_id
    
    cat("\nProcessing metric:", metric, "\n")
    cat("Number of species before NA removal:", nrow(shark.info), "\n")
    cat("Number of species after NA removal:", nrow(info), "\n")
    
    transformed <- info %>%
      dplyr::select(species_id, all_of(metric)) %>%
      dplyr::rename(sp = species_id, 
                    amount = !!sym(metric))
    
    # The values should already be normalized from earlier processing
    # but we'll check and normalize if needed
    if(max(transformed$amount, na.rm = TRUE) > 1) {
      transformed <- transformed %>%
        dplyr::mutate(amount = amount / max(amount, na.rm = TRUE))
    }
    
    cat("Summary of transformed values:\n")
    print(summary(transformed$amount))
    
    return(list(
      info = info,
      species = species_ids,
      transformed = transformed,
      species_names = info$Species
    ))
  }
  
  metrics <- c("EDGE_P100", "FUSE", "EDGE2", "IUCN", "ED", "FSp", "FUn")
  results <- lapply(metrics, function(metric) {
    process_dataset(zone_shark_info, puvsp, metric)
  })
  
  names(results) <- metrics
  
  # Save results
  output_data <- list(
    metrics = results,
    species_lookup = species_lookup[species_lookup$species_name %in% species_in_zone, ]
  )
  
  saveRDS(output_data, 
          here::here("Data", "My dataframes", 
                     paste0(zone_name, "_shark_conservation_metrics.rds")))
  
  jsonlite::write_json(output_data,
                       here::here("Data", "My dataframes", 
                                  paste0(zone_name, "_shark_conservation_metrics.json")))
  
  return(results)
}

# Filter both PUVSP datasets
highseas_species <- filter_puvsp_to_metrics(highseas_species, shark.info_complete, global_species_lookup)
continental_species <- filter_puvsp_to_metrics(continental_species, shark.info_complete, global_species_lookup)

#You've lost ten species, find which ones:
# Get the species from shark.info_complete
shark_species <- shark.info_complete$Species

# Get the species from your presence data (excluding the first 3 columns which are id, lon, lat)
presence_species <- names(highseas_species)[4:ncol(highseas_species)]

# Find species that are in shark.info_complete but not in presence data
missing_species <- setdiff(shark_species, presence_species)

# Show the missing species
print("Species with metrics but no presence data:")
print(missing_species)

# Process for high seas
highseas_results <- process_zone_metrics(shark.info_complete, 
                                         highseas_species, 
                                         "highseas",
                                         global_species_lookup)

# Process for continental waters
continental_results <- process_zone_metrics(shark.info_complete, 
                                            continental_species, 
                                            "continental",
                                            global_species_lookup)
# Add the MPA data ----
# Function to read and filter MPA data
read_and_filter_mpa <- function(file_number) {
  file_path <- here("Data", "Raw", "WDPA_WDOECM_Sep2024_Public_marine_shp", 
                    paste0("WDPA_WDOECM_Sep2024_Public_marine_shp_", file_number),
                    "WDPA_WDOECM_Sep2024_Public_marine_shp-polygons.shp")
  
  tryCatch({
    mpa <- st_read(file_path)
    mpa_filtered <- mpa %>% filter(IUCN_CAT %in% c("Ia", "Ib", "II", "III"))
    list(all = mpa, filtered = mpa_filtered)
  }, error = function(e) {
    warning(paste("Error reading file", file_path, ":", e$message))
    NULL
  })
}

# Read and filter MPA data
mpa_data <- map(0:2, read_and_filter_mpa)

# Combine MPA data
mpa_NT <- do.call(rbind, map(mpa_data, "filtered"))
mpa_all <- do.call(rbind, map(mpa_data, "all"))

# Reset row names
row.names(mpa_NT) <- NULL
row.names(mpa_all) <- NULL

# Create raster layout for MPA data
rast <- raster::raster(xmn=-180.25, xmx=180.75, ymn=-90.25, ymx=90.75, resolution=c(0.5,0.5))

# Rasterize MPA data
mpa_NT_raster <- rasterize(st_sf(mpa_NT$geometry), rast)
mpa_ALL_raster <- rasterize(st_sf(mpa_all$geometry), rast)

# Modified process_raster function
process_raster <- function(raster_data, coords_df) {
  # Convert coordinates to cell numbers
  cells <- cellFromXY(raster_data, 
                      xy = data.frame(x = coords_df$Coord_x, 
                                      y = coords_df$Coord_y))
  
  # Extract values for just those cells
  values <- raster::extract(raster_data, cells)
  
  # Create dataframe with the correct IDs
  df <- data.frame(
    id = coords_df$id,
    status = ifelse(is.na(values), 0, 1)
  )
  
  return(df)
}

# Process separately for high seas and continental waters
mpa_NT_df_highseas <- process_raster(mpa_NT_raster, highseas_coordinates)
mpa_NT_df_continental <- process_raster(mpa_NT_raster, continental_coordinates)

mpa_ALL_df_highseas <- process_raster(mpa_ALL_raster, highseas_coordinates)
mpa_ALL_df_continental <- process_raster(mpa_ALL_raster, continental_coordinates)

# Process for high seas
highseas_fishing_notake <- dplyr::left_join(highseas_fishing_final, mpa_NT_df_highseas, by = "id") %>%
  dplyr::select(id, status, cost)

highseas_fishing_allMPAs <- dplyr::left_join(highseas_fishing_final, mpa_ALL_df_highseas, by = "id") %>%
  dplyr::select(id, status, cost)

# Process for continental waters
continental_fishing_notake <- dplyr::left_join(continental_fishing_final, mpa_NT_df_continental, by = "id") %>%
  dplyr::select(id, status, cost)

continental_fishing_allMPAs <- dplyr::left_join(continental_fishing_final, mpa_ALL_df_continental, by = "id") %>%
  dplyr::select(id, status, cost)

# Save results for high seas
readr::write_csv(highseas_fishing_notake, 
                 here("Data", "My dataframes", "highseas_fishing_notake.csv"))
readr::write_csv(highseas_fishing_allMPAs, 
                 here("Data", "My dataframes", "highseas_fishing_allMPAs.csv"))

# Save results for continental waters 
readr::write_csv(continental_fishing_notake, 
                 here("Data", "My dataframes", "continental_fishing_notake.csv"))
readr::write_csv(continental_fishing_allMPAs, 
                 here("Data", "My dataframes", "continental_fishing_allMPAs.csv"))

# Create binary rasters from the existing rasters
mpa_NT_binary <- mpa_NT_raster
mpa_NT_binary[!is.na(mpa_NT_binary)] <- 1
mpa_NT_binary[is.na(mpa_NT_binary)] <- 0

mpa_ALL_binary <- mpa_ALL_raster
mpa_ALL_binary[!is.na(mpa_ALL_binary)] <- 1
mpa_ALL_binary[is.na(mpa_ALL_binary)] <- 0

# Plot the binary rasters
plot(mpa_NT_binary, main="No-Take MPAs")
plot(mpa_ALL_binary, main="All MPAs")

# Create masks using coordinates
create_zone_mask <- function(rast_template, coords_df) {
  # Create empty raster with same properties as template
  zone_mask <- rast_template
  zone_mask[] <- 0
  
  # Convert coordinates to cell numbers in the raster
  cells <- terra::cellFromXY(zone_mask, 
                             xy = data.frame(x = coords_df$Coord_x, 
                                             y = coords_df$Coord_y))
  
  # Set those cells to 1
  zone_mask[cells] <- 1
  
  return(zone_mask)
}

# Create masks for each zone using the coordinates
highseas_mask <- create_zone_mask(rast, highseas_coordinates)
continental_mask <- create_zone_mask(rast, continental_coordinates)

# Create binary rasters for high seas
highseas_NT_binary <- mpa_NT_raster * highseas_mask
highseas_ALL_binary <- mpa_ALL_raster * highseas_mask

# Create binary rasters for continental waters
continental_NT_binary <- mpa_NT_raster * continental_mask
continental_ALL_binary <- mpa_ALL_raster * continental_mask

# Convert to binary (0/1)
highseas_NT_binary[highseas_NT_binary > 0] <- 1
highseas_NT_binary[is.na(highseas_NT_binary)] <- 0

highseas_ALL_binary[highseas_ALL_binary > 0] <- 1
highseas_ALL_binary[is.na(highseas_ALL_binary)] <- 0

continental_NT_binary[continental_NT_binary > 0] <- 1
continental_NT_binary[is.na(continental_NT_binary)] <- 0

continental_ALL_binary[continental_ALL_binary > 0] <- 1
continental_ALL_binary[is.na(continental_ALL_binary)] <- 0

# Plot to verify
par(mfrow=c(2,2))
plot(highseas_NT_binary, main="High Seas No-Take MPAs")
plot(highseas_ALL_binary, main="High Seas All MPAs")
plot(continental_NT_binary, main="Continental No-Take MPAs")
plot(continental_ALL_binary, main="Continental All MPAs")
par(mfrow=c(1,1))

# Save the binary rasters
writeRaster(highseas_NT_binary, 
            filename=here::here("Data", "My dataframes", "highseas_mpa_NT_binary.tif"), 
            format="GTiff", 
            overwrite=TRUE)
writeRaster(highseas_ALL_binary, 
            filename=here::here("Data", "My dataframes", "highseas_mpa_ALL_binary.tif"), 
            format="GTiff", 
            overwrite=TRUE)

writeRaster(continental_NT_binary, 
            filename=here::here("Data", "My dataframes", "continental_mpa_NT_binary.tif"), 
            format="GTiff", 
            overwrite=TRUE)
writeRaster(continental_ALL_binary, 
            filename=here::here("Data", "My dataframes", "continental_mpa_ALL_binary.tif"), 
            format="GTiff", 
            overwrite=TRUE)

# Harmonise the number of species in all datasets: puvsp and shark_conservation_metric ----
library(dplyr)
library(tidyr)
library(purrr)
library(stringr)
library(here)

# Load the existing files and global species lookup
highseas_data <- readRDS(here::here("Data", "My dataframes", "highseas_shark_conservation_metrics.rds"))
continental_data <- readRDS(here::here("Data", "My dataframes", "continental_shark_conservation_metrics.rds"))
global_species_lookup <- readRDS(here::here("Data", "My dataframes", "global_species_lookup.rds"))

# Extract metrics and lookup from the saved data
highseas_metrics <- highseas_data$metrics
continental_metrics <- continental_data$metrics

# Function to harmonize metrics with PUVSP species
harmonize_data <- function(metrics_data, puvsp_data, zone_name, species_lookup) {
  # Print initial counts
  cat("\nInitial counts for", zone_name, ":\n")
  cat("Metrics species:", length(unique(metrics_data$EDGE_P100$transformed$sp)), "\n")
  cat("PUVSP species:", length(unique(puvsp_data$sp)), "\n")
  
  # Get metrics species IDs
  metrics_species_ids <- unique(metrics_data$EDGE_P100$transformed$sp)
  
  # First, harmonize PUVSP to only include species that exist in metrics
  harmonized_puvsp <- puvsp_data %>%
    filter(sp %in% metrics_species_ids)
  
  # Get the final list of matching species
  final_species_ids <- unique(harmonized_puvsp$sp)
  
  cat("Matching species after PUVSP harmonization:", length(final_species_ids), "\n")
  
  # Then harmonize metrics to match the filtered PUVSP species
  harmonized_metrics <- lapply(metrics_data, function(metric) {
    metric$transformed <- metric$transformed %>%
      filter(sp %in% final_species_ids)
    
    metric$species <- metric$species[metric$species %in% final_species_ids]
    metric$info <- metric$info %>%
      filter(species_id %in% final_species_ids)
    
    # Add species names to transformed data
    metric$transformed <- metric$transformed %>%
      left_join(species_lookup, by = c("sp" = "species_id"))
    
    return(metric)
  })
  
  # Verify final counts
  cat("Final counts:\n")
  cat("Harmonized metrics species:", length(unique(harmonized_metrics$EDGE_P100$transformed$sp)), "\n")
  cat("Harmonized PUVSP species:", length(unique(harmonized_puvsp$sp)), "\n")
  
  # Create harmonized data object
  harmonized_data <- list(
    metrics = harmonized_metrics,
    puvsp = harmonized_puvsp,
    species_lookup = species_lookup %>%
      filter(species_id %in% final_species_ids)
  )
  
  # Save harmonized data
  saveRDS(harmonized_data, 
          here::here("Data", "My dataframes", 
                     paste0(zone_name, "_shark_conservation_metrics_harmonised.rds")))
  
  # Save harmonized PUVSP with species names
  harmonized_puvsp %>%
    left_join(species_lookup, by = c("sp" = "species_id")) %>%
    write.csv(here::here("Data", "My dataframes", 
                         paste0(zone_name, "_puvsp_harmonised.csv")),
              row.names = FALSE)
  
  # Return summary
  species_summary <- species_lookup %>%
    filter(species_id %in% final_species_ids)
  
  return(list(
    n_species = nrow(species_summary),
    species_ids = species_summary$species_id,
    species_names = species_summary$species_name
  ))
}

# Re-run the harmonization
highseas_harmonized <- harmonize_data(
  highseas_metrics,
  highseas_puvsp,
  "highseas",
  global_species_lookup
)

# Harmonize continental data
continental_harmonized <- harmonize_data(
  continental_metrics,
  continental_puvsp,
  "continental",
  global_species_lookup
)

# Print detailed summaries
cat("\nHigh Seas Harmonized Summary:\n")
cat("Number of species:", highseas_harmonized$n_species, "\n")
cat("Species names:\n")
print(data.frame(
  ID = highseas_harmonized$species_ids,
  Species = highseas_harmonized$species_names
))

cat("\nContinental Waters Harmonized Summary:\n")
cat("Number of species:", continental_harmonized$n_species, "\n")
cat("Species names:\n")
print(data.frame(
  ID = continental_harmonized$species_ids,
  Species = continental_harmonized$species_names
))

# Verify that species match between metrics and PUVSP for each zone
verify_species_match <- function(zone_name) {
  # Load harmonized data
  metrics <- readRDS(here::here("Data", "My dataframes", 
                                paste0(zone_name, "_shark_conservation_metrics_harmonised.rds")))
  puvsp <- read.csv(here::here("Data", "My dataframes", 
                               paste0(zone_name, "_puvsp_harmonised.csv")))
  
  # Access species IDs from the harmonized metrics
  metrics_species <- unique(metrics$metrics$EDGE_P100$transformed$sp)
  puvsp_species <- unique(puvsp$sp)
  
  cat("\nVerification for", zone_name, ":\n")
  cat("Species in harmonized metrics:", length(metrics_species), "\n")
  cat("Species in harmonized PUVSP:", length(puvsp_species), "\n")
  cat("Species in metrics (first few):", paste(head(metrics_species), collapse=", "), "\n")
  cat("Species in PUVSP (first few):", paste(head(puvsp_species), collapse=", "), "\n")
  cat("Matching species:", length(intersect(metrics_species, puvsp_species)), "\n")
  
  return(list(
    metrics_species = metrics_species,
    puvsp_species = puvsp_species
  ))
}

# Run verification on harmonized data
verify_species_match("highseas")
verify_species_match("continental")

# Bind conservation metrics to a minimum of 10% and 30% ----

create_budget_scenarios <- function(zone_name) {
  # Load harmonized metrics from RDS file
  metrics <- readRDS(here::here("Data", "My dataframes", 
                                paste0(zone_name, "_shark_conservation_metrics_harmonised.rds")))
  
  # Extract just the metrics part from the loaded data
  metrics <- metrics$metrics
  
  # Create 10% scenario
  metrics_10 <- metrics
  for(metric_name in names(metrics_10)) {
    metrics_10[[metric_name]]$transformed$amount <- pmax(metrics_10[[metric_name]]$transformed$amount, 0.1)
  }
  
  # Create 30% scenario
  metrics_30 <- metrics
  for(metric_name in names(metrics_30)) {
    metrics_30[[metric_name]]$transformed$amount <- pmax(metrics_30[[metric_name]]$transformed$amount, 0.3)
  }
  
  # Save as RDS files
  saveRDS(metrics_10, here::here("Data", "My dataframes", 
                                 paste0(zone_name, "_shark_conservation_metrics_10_harmonised.rds")))
  saveRDS(metrics_30, here::here("Data", "My dataframes", 
                                 paste0(zone_name, "_shark_conservation_metrics_30_harmonised.rds")))
  
  # Also save as JSON if needed
  jsonlite::write_json(metrics_10, here::here("Data", "My dataframes", 
                                              paste0(zone_name, "_shark_conservation_metrics_10_harmonised.json")))
  jsonlite::write_json(metrics_30, here::here("Data", "My dataframes", 
                                              paste0(zone_name, "_shark_conservation_metrics_30_harmonised.json")))
}

# Create scenarios for both zones
create_budget_scenarios("continental")
create_budget_scenarios("highseas")

check_metrics <- function(zone_name) {
  metrics_10 <- jsonlite::fromJSON(here::here("Data", "My dataframes", 
                                              paste0(zone_name, "_shark_conservation_metrics_10.json")),
                                   simplifyDataFrame = TRUE)
  metrics_30 <- jsonlite::fromJSON(here::here("Data", "My dataframes", 
                                              paste0(zone_name, "_shark_conservation_metrics_30.json")),
                                   simplifyDataFrame = TRUE)
  
  # Function to get summary statistics for each metric
  summarize_metric <- function(metric_data) {
    # Convert to numeric if necessary and handle NAs
    amount <- as.numeric(metric_data$transformed$amount)
    amount <- amount[!is.na(amount)]
    
    return(data.frame(
      min = min(amount),
      median = median(amount),
      mean = round(mean(amount), 4),
      max = max(amount),
      n_below_threshold = sum(amount < 0.1)
    ))
  }
  
  # Create summary tables
  cat("\nSummary for", zone_name, ":\n")
  
  cat("\n10% Budget Scenario:\n")
  summary_10 <- do.call(rbind, lapply(names(metrics_10), function(metric) {
    stats <- summarize_metric(metrics_10[[metric]])
    cbind(metric = metric, stats)
  }))
  rownames(summary_10) <- NULL
  print(summary_10)
  
  cat("\n30% Budget Scenario:\n")
  summary_30 <- do.call(rbind, lapply(names(metrics_30), function(metric) {
    stats <- summarize_metric(metrics_30[[metric]])
    cbind(metric = metric, stats)
  }))
  rownames(summary_30) <- NULL
  print(summary_30)
}

# Check metrics
check_metrics("continental")
check_metrics("highseas")

# Check all datasets ----
pu_fishing_data <- read.csv(here::here("Data", 
                                       "My Dataframes", 
                                       "Continental", 
                                       "continental_fishing_allMPAs.csv"))
summary(pu_fishing_data)

puvsp <- read.csv(here::here("Data", 
                             "My Dataframes", 
                             "Continental", 
                             "continental_puvsp_harmonised.csv"))
summary(puvsp)
length(unique(puvsp$sp))

pu_coords <- read.csv(here::here("Data", 
                                 "My Dataframes", 
                                 "Continental", 
                                 "continental_coords_harmonised.csv"))
summary(pu_coords)

shark_conservation_metrics <- jsonlite::fromJSON(here::here("Data", "My dataframes", #"Continental", 
                                                            "continental_shark_conservation_metrics_10_harmonised.json"))

# View the structure of the data
str(shark_conservation_metrics)
length(shark_conservation_metrics$EDGE2$transformed[,1])
summary(as.data.frame(shark_conservation_metrics$metrics$EDGE2$transformed))

summary(shark_conservation_metrics$IUCN$info)

#
pu_fishing_data <- read.csv(here::here("Data", 
                                       "My Dataframes", 
                                       "High Seas", 
                                       "highseas_fishing_allMPAs.csv"))
summary(pu_fishing_data)

puvsp <- read.csv(here::here("Data", 
                             "My Dataframes", 
                             #"High Seas", 
                             "highseas_puvsp_harmonised.csv"))
summary(puvsp)
length(unique(puvsp$sp))
pu_coords <- read.csv(here::here("Data", 
                                 "My Dataframes", 
                                 "High Seas", 
                                 "highseas_pu_coords.csv"))
summary(pu_coords)

shark_conservation_metrics <- jsonlite::fromJSON(here::here("Data", "My dataframes", "High Seas", 
                                                            "highseas_shark_conservation_metrics_30_harmonised.json"))

# View the structure of the data
str(shark_conservation_metrics)
length(shark_conservation_metrics$metrics$EDGE2$transformed[,1])
summary(as.data.frame(shark_conservation_metrics$EDGE2$transformed))

# Add IUCN probability metrics to conservation metrics ----

add_iucn_metrics <- function(zone_name) {
  # Load harmonized metrics from RDS file for both scenarios
  metrics_10 <- readRDS(here::here("Data", "My dataframes", 
                                   paste0(zone_name, "_shark_conservation_metrics_10_harmonised.rds")))
  metrics_30 <- readRDS(here::here("Data", "My dataframes", 
                                   paste0(zone_name, "_shark_conservation_metrics_30_harmonised.rds")))
  
  # Load IUCN data
  IUCN_final <- readRDS(here::here("Data", "GAP analyses", "sharks_iucn_final.rds"))
  
  # Preprocess the IUCN data: replace underscore with space in species names
  IUCN_final <- IUCN_final %>%
    dplyr::mutate(Species = gsub("_", " ", Species))

  # 32 species are named differently in the range data and in sharks_iucn_final (genus, spelling or gender ending):
  # use the range data names so that they are matched instead of being assigned the most common status
  iucn_name_corrections <- read.csv(here::here("Data", "My dataframes", "iucn_name_corrections.csv"))
  IUCN_final <- IUCN_final %>%
    dplyr::mutate(Species = dplyr::coalesce(iucn_name_corrections$range_name[match(Species, iucn_name_corrections$iucn_name)], Species))
  
  # Load species lookup for converting between names and IDs
  global_species_lookup <- readRDS(here::here("Data", "My dataframes", "global_species_lookup.rds"))
  
  # Process each IUCN metric and add it to the metrics structure
  process_iucn_metric <- function(metrics, metric_name) {
    # Extract data from IUCN_final
    iucn_data <- IUCN_final %>%
      dplyr::select(Species, !!sym(metric_name)) %>%
      dplyr::rename(amount = !!sym(metric_name)) %>%
      dplyr::left_join(global_species_lookup, by = c("Species" = "species_name")) %>%
      dplyr::select(sp = species_id, amount)  # Removed species_name here
    
    # For each metric set, only keep species that are in the harmonized datasets
    existing_species <- unique(metrics$EDGE_P100$transformed$sp)
    iucn_filtered <- iucn_data %>%
      dplyr::filter(sp %in% existing_species)
    
    # Apply 10% or 30% threshold based on the function argument name
    if(deparse(substitute(metrics)) == "metrics_10") {
      iucn_filtered$amount <- pmax(iucn_filtered$amount, 0.1)
    } else if(deparse(substitute(metrics)) == "metrics_30") {
      iucn_filtered$amount <- pmax(iucn_filtered$amount, 0.3)
    }
    
    # Create the metric structure
    new_metric <- list(
      transformed = iucn_filtered,
      species = unique(iucn_filtered$sp),
      info = global_species_lookup %>% 
        dplyr::filter(species_id %in% unique(iucn_filtered$sp)) %>%
        dplyr::mutate(!!metric_name := iucn_filtered$amount[match(species_id, iucn_filtered$sp)])
    )
    
    return(new_metric)
  }
  
  # Add each IUCN metric to metrics_10
  metrics_10$iucn_GE <- process_iucn_metric(metrics_10, "iucn_GE")
  metrics_10$iucn_P50 <- process_iucn_metric(metrics_10, "iucn_P50")
  metrics_10$iucn_P100 <- process_iucn_metric(metrics_10, "iucn_P100")
  
  # Add each IUCN metric to metrics_30
  metrics_30$iucn_GE <- process_iucn_metric(metrics_30, "iucn_GE")
  metrics_30$iucn_P50 <- process_iucn_metric(metrics_30, "iucn_P50")
  metrics_30$iucn_P100 <- process_iucn_metric(metrics_30, "iucn_P100")
  
  # Normalize iucn_GE values (since they're categorical 0-4)
  metrics_10$iucn_GE$transformed$amount <- metrics_10$iucn_GE$transformed$amount / 4
  metrics_30$iucn_GE$transformed$amount <- metrics_30$iucn_GE$transformed$amount / 4
  
  # Save updated files
  saveRDS(metrics_10, here::here("Data", "My dataframes", 
                                 paste0(zone_name, "_shark_conservation_metrics_10_harmonised.rds")))
  saveRDS(metrics_30, here::here("Data", "My dataframes", 
                                 paste0(zone_name, "_shark_conservation_metrics_30_harmonised.rds")))
  
  # Save as JSON
  jsonlite::write_json(metrics_10, here::here("Data", "My dataframes", 
                                              paste0(zone_name, "_shark_conservation_metrics_10_harmonised.json")))
  jsonlite::write_json(metrics_30, here::here("Data", "My dataframes", 
                                              paste0(zone_name, "_shark_conservation_metrics_30_harmonised.json")))
  
  # Return counts for verification
  return(list(
    scenario_10 = list(
      total_species = length(unique(metrics_10$EDGE_P100$transformed$sp)),
      iucn_GE_species = length(metrics_10$iucn_GE$species),
      iucn_P50_species = length(metrics_10$iucn_P50$species),
      iucn_P100_species = length(metrics_10$iucn_P100$species)
    ),
    scenario_30 = list(
      total_species = length(unique(metrics_30$EDGE_P100$transformed$sp)),
      iucn_GE_species = length(metrics_30$iucn_GE$species),
      iucn_P50_species = length(metrics_30$iucn_P50$species),
      iucn_P100_species = length(metrics_30$iucn_P100$species)
    )
  ))
}

# Apply to both zones
continental_results <- add_iucn_metrics("continental")
highseas_results <- add_iucn_metrics("highseas")

# Print verification
print("Continental Waters Results:")
print(continental_results)

print("High Seas Results:")
print(highseas_results)

# Create a diagnostic report to identify which species are missing 

identify_missing_species <- function(zone_name) {
  # Load necessary data
  metrics <- readRDS(here::here("Data", "My dataframes", 
                                paste0(zone_name, "_shark_conservation_metrics_10_harmonised.rds")))
  IUCN_final <- readRDS(here::here("Data", "GAP analyses", "sharks_iucn_final.rds"))
  IUCN_final$Species <- gsub("_", " ", IUCN_final$Species)
  
  # Get species from metrics
  metrics_species <- unique(metrics$EDGE_P100$transformed$sp)
  
  # Get global species lookup
  global_species_lookup <- readRDS(here::here("Data", "My dataframes", "global_species_lookup.rds"))
  
  # Get species names that are in metrics
  metrics_species_names <- global_species_lookup$species_name[global_species_lookup$species_id %in% metrics_species]
  
  # Get species names from IUCN
  iucn_species <- unique(IUCN_final$Species)
  
  # Find missing species
  missing_species <- setdiff(metrics_species_names, iucn_species)
  
  # Create diagnostic dataframe
  diagnostics <- data.frame(
    species_name = missing_species,
    species_id = global_species_lookup$species_id[
      match(missing_species, global_species_lookup$species_name)
    ]
  )
  
  # Save diagnostics
  write.csv(diagnostics, 
            here::here("Data", "My dataframes", 
                       paste0(zone_name, "_missing_iucn_species.csv")),
            row.names = FALSE)
  
  return(diagnostics)
}

# Run diagnostics
continental_missing <- identify_missing_species("continental")
highseas_missing <- identify_missing_species("highseas")

# Print first few missing species for each zone
cat("First few missing species in Continental Waters:\n")
print(head(continental_missing))

cat("\nFirst few missing species in High Seas:\n")
print(head(highseas_missing))

# Add IUCN probability metrics to conservation metrics with imputation
add_iucn_metrics_with_imputation <- function(zone_name) {
  # Load harmonized metrics from RDS file for both scenarios
  metrics_10 <- readRDS(here::here("Data", "My dataframes", 
                                   paste0(zone_name, "_shark_conservation_metrics_10_harmonised.rds")))
  metrics_30 <- readRDS(here::here("Data", "My dataframes", 
                                   paste0(zone_name, "_shark_conservation_metrics_30_harmonised.rds")))
  
  # Load IUCN data
  IUCN_final <- readRDS(here::here("Data", "GAP analyses", "sharks_iucn_final.rds"))
  
  # Preprocess the IUCN data: replace underscore with space in species names
  IUCN_final <- IUCN_final %>%
    dplyr::mutate(Species = gsub("_", " ", Species))

  # 32 species are named differently in the range data and in sharks_iucn_final (genus, spelling or gender ending):
  # use the range data names so that they are matched instead of being assigned the most common status
  iucn_name_corrections <- read.csv(here::here("Data", "My dataframes", "iucn_name_corrections.csv"))
  IUCN_final <- IUCN_final %>%
    dplyr::mutate(Species = dplyr::coalesce(iucn_name_corrections$range_name[match(Species, iucn_name_corrections$iucn_name)], Species))
  
  # Load species lookup for converting between names and IDs
  global_species_lookup <- readRDS(here::here("Data", "My dataframes", "global_species_lookup.rds"))
  
  # Extract species names for all species in the metrics
  metrics_species <- unique(metrics_10$EDGE_P100$transformed$sp)
  metrics_species_names <- global_species_lookup$species_name[
    match(metrics_species, global_species_lookup$species_id)
  ]
  
  # Find missing species
  missing_species <- data.frame(
    species_id = metrics_species[!metrics_species %in% 
                                   global_species_lookup$species_id[
                                     global_species_lookup$species_name %in% IUCN_final$Species
                                   ]]
  )
  
  # Join with species lookup to get names
  missing_species <- missing_species %>%
    left_join(global_species_lookup, by = c("species_id" = "species_id")) %>%
    select(species_id, species_name)
  
  # Log missing species
  write.csv(missing_species, 
            here::here("Data", "My dataframes", 
                       paste0(zone_name, "_missing_iucn_species.csv")),
            row.names = FALSE)
  
  cat("Number of species missing IUCN data for", zone_name, ":", nrow(missing_species), "out of", length(metrics_species), "\n")
  
# Process each IUCN metric and add it to the metrics structure
process_iucn_metric <- function(metrics, metric_name, threshold) {
    # Extract data from IUCN_final
    iucn_data <- IUCN_final %>%
      dplyr::select(Species, !!sym(metric_name)) %>%
      dplyr::rename(amount = !!sym(metric_name)) %>%
      dplyr::left_join(global_species_lookup, by = c("Species" = "species_name")) %>%
      dplyr::select(sp = species_id, amount)
    
    # Get all species from metrics
    existing_species <- unique(metrics$EDGE_P100$transformed$sp)
    
    # Create a complete dataframe with all species, imputing NA for missing ones
    complete_data <- data.frame(
      sp = existing_species
    ) %>%
      dplyr::left_join(iucn_data, by = "sp")
    
    # Imputation strategy
    if(metric_name == "iucn_GE") {
      # For categorical data, use mode (most common value)
      mode_value <- as.numeric(names(sort(table(complete_data$amount), decreasing = TRUE)[1]))
      if(is.na(mode_value)) mode_value <- 2  # Default to "2" if mode is NA
      complete_data$amount[is.na(complete_data$amount)] <- mode_value
    } else {
      # For probability data, use median
      median_value <- median(complete_data$amount, na.rm = TRUE)
      if(is.na(median_value)) median_value <- 0.5  # Default to 0.5 if median is NA
      complete_data$amount[is.na(complete_data$amount)] <- median_value
    }
    
    # Apply threshold
    complete_data$amount <- pmax(complete_data$amount, threshold)
    
    # For iucn_GE, normalize to 0-1 range
    if(metric_name == "iucn_GE") {
      complete_data$amount <- complete_data$amount / 4
    }
    
    # Create the metric structure
    new_metric <- list(
      transformed = complete_data,
      species = existing_species,
      info = global_species_lookup %>% 
        dplyr::filter(species_id %in% existing_species) %>%
        dplyr::mutate(!!metric_name := complete_data$amount[match(species_id, complete_data$sp)])
    )
    
    return(new_metric)
  }
  
  # Add each IUCN metric to metrics_10
  metrics_10$iucn_GE <- process_iucn_metric(metrics_10, "iucn_GE", 0.1)
  metrics_10$iucn_P50 <- process_iucn_metric(metrics_10, "iucn_P50", 0.1)
  metrics_10$iucn_P100 <- process_iucn_metric(metrics_10, "iucn_P100", 0.1)
  
  # Add each IUCN metric to metrics_30
  metrics_30$iucn_GE <- process_iucn_metric(metrics_30, "iucn_GE", 0.3)
  metrics_30$iucn_P50 <- process_iucn_metric(metrics_30, "iucn_P50", 0.3)
  metrics_30$iucn_P100 <- process_iucn_metric(metrics_30, "iucn_P100", 0.3)
  
  # Save updated files with "_IUCN" in the filenames
  saveRDS(metrics_10, here::here("Data", "My dataframes", 
                                 paste0(zone_name, "_shark_conservation_metrics_10_harmonised_IUCN.rds")))
  saveRDS(metrics_30, here::here("Data", "My dataframes", 
                                 paste0(zone_name, "_shark_conservation_metrics_30_harmonised_IUCN.rds")))
  
  # Save as JSON with "_IUCN" in the filenames
  jsonlite::write_json(metrics_10, here::here("Data", "My dataframes", 
                                              paste0(zone_name, "_shark_conservation_metrics_10_harmonised_IUCN.json")))
  jsonlite::write_json(metrics_30, here::here("Data", "My dataframes", 
                                              paste0(zone_name, "_shark_conservation_metrics_30_harmonised_IUCN.json")))
  
  # Return counts for verification
  return(list(
    scenario_10 = list(
      total_species = length(unique(metrics_10$EDGE_P100$transformed$sp)),
      iucn_GE_species = length(metrics_10$iucn_GE$species),
      iucn_P50_species = length(metrics_10$iucn_P50$species),
      iucn_P100_species = length(metrics_10$iucn_P100$species)
    ),
    scenario_30 = list(
      total_species = length(unique(metrics_30$EDGE_P100$transformed$sp)),
      iucn_GE_species = length(metrics_30$iucn_GE$species),
      iucn_P50_species = length(metrics_30$iucn_P50$species),
      iucn_P100_species = length(metrics_30$iucn_P100$species)
    )
  ))
}

# Apply to both zones
continental_results <- add_iucn_metrics_with_imputation("continental")
highseas_results <- add_iucn_metrics_with_imputation("highseas")

# Print verification
print("Continental Waters Results:")
print(continental_results)

print("High Seas Results:")
print(highseas_results)

# Function to summarize metrics datasets
summarize_metrics <- function(zone_name) {
  # Load the saved files
  metrics_10_iucn <- readRDS(here::here("Data", "My dataframes", 
                                        paste0(zone_name, "_shark_conservation_metrics_10_harmonised_IUCN.rds")))
  metrics_30_iucn <- readRDS(here::here("Data", "My dataframes", 
                                        paste0(zone_name, "_shark_conservation_metrics_30_harmonised_IUCN.rds")))
  
  # Function to summarize a single metric
  summarize_one_metric <- function(metric, metric_name) {
    if(is.null(metric) || !is.list(metric) || !("transformed" %in% names(metric))) {
      return(paste("Metric", metric_name, "is missing or malformed"))
    }
    
    data <- metric$transformed$amount
    
    if(length(data) == 0) {
      return(paste("No data for", metric_name))
    }
    
    summary_stats <- list(
      metric = metric_name,
      n = length(data),
      min = min(data, na.rm = TRUE),
      max = max(data, na.rm = TRUE),
      mean = mean(data, na.rm = TRUE),
      median = median(data, na.rm = TRUE),
      na_count = sum(is.na(data))
    )
    
    return(summary_stats)
  }
  
  # Get all metric names
  metric_names_10 <- names(metrics_10_iucn)
  metric_names_30 <- names(metrics_30_iucn)
  
  # Create summaries for 10% scenario
  summaries_10 <- lapply(metric_names_10, function(name) {
    summarize_one_metric(metrics_10_iucn[[name]], name)
  })
  names(summaries_10) <- metric_names_10
  
  # Create summaries for 30% scenario
  summaries_30 <- lapply(metric_names_30, function(name) {
    summarize_one_metric(metrics_30_iucn[[name]], name)
  })
  names(summaries_30) <- metric_names_30
  
  # Special focus on IUCN metrics
  cat("\n======= IUCN Metrics for", zone_name, "10% scenario =======\n")
  if("iucn_GE" %in% metric_names_10) {
    print(summarize_one_metric(metrics_10_iucn$iucn_GE, "iucn_GE"))
    cat("\nValue distribution for iucn_GE:\n")
    print(table(metrics_10_iucn$iucn_GE$transformed$amount))
  }
  
  if("iucn_P50" %in% metric_names_10) {
    print(summarize_one_metric(metrics_10_iucn$iucn_P50, "iucn_P50"))
  }
  
  if("iucn_P100" %in% metric_names_10) {
    print(summarize_one_metric(metrics_10_iucn$iucn_P100, "iucn_P100"))
  }
  
  cat("\n======= IUCN Metrics for", zone_name, "30% scenario =======\n")
  if("iucn_GE" %in% metric_names_30) {
    print(summarize_one_metric(metrics_30_iucn$iucn_GE, "iucn_GE"))
  }
  
  if("iucn_P50" %in% metric_names_30) {
    print(summarize_one_metric(metrics_30_iucn$iucn_P50, "iucn_P50"))
  }
  
  if("iucn_P100" %in% metric_names_30) {
    print(summarize_one_metric(metrics_30_iucn$iucn_P100, "iucn_P100"))
  }
  
  # Check structure consistency
  cat("\n======= Structure check for", zone_name, "=======\n")
  cat("Number of species in 10% scenario:", length(metrics_10_iucn$EDGE_P100$species), "\n")
  cat("Number of species in 30% scenario:", length(metrics_30_iucn$EDGE_P100$species), "\n")
  
  # Verify JSON files by loading them back
  json_10 <- jsonlite::read_json(here::here("Data", "My dataframes", 
                                            paste0(zone_name, "_shark_conservation_metrics_10_harmonised_IUCN.json")))
  json_30 <- jsonlite::read_json(here::here("Data", "My dataframes", 
                                            paste0(zone_name, "_shark_conservation_metrics_30_harmonised_IUCN.json")))
  
  cat("\nJSON files successfully loaded\n")
  cat("Metrics in 10% JSON:", paste(names(json_10), collapse=", "), "\n")
  cat("Metrics in 30% JSON:", paste(names(json_30), collapse=", "), "\n")
  
  return(list(
    metrics_10_summary = summaries_10,
    metrics_30_summary = summaries_30
  ))
}

# Summarize the continental and highseas datasets
continental_summaries <- summarize_metrics("continental")
highseas_summaries <- summarize_metrics("highseas")

# Print out some key information from the summaries
cat("\n====== CONTINENTAL SUMMARIES ======\n")
cat("10% scenario IUCN metrics:\n")
print(continental_summaries$metrics_10_summary$iucn_GE)
print(continental_summaries$metrics_10_summary$iucn_P50)
print(continental_summaries$metrics_10_summary$iucn_P100)

cat("\n====== HIGHSEAS SUMMARIES ======\n")
cat("10% scenario IUCN metrics:\n")
print(highseas_summaries$metrics_10_summary$iucn_GE)
print(highseas_summaries$metrics_10_summary$iucn_P50)
print(highseas_summaries$metrics_10_summary$iucn_P100)

# Create individual .tif files (continental) ----
library(raster)
library(dplyr)
library(tidyr)
library(here)

# Create directory if it doesn't exist
tif_dir <- here::here("Data", "tif files continental")
if (!dir.exists(tif_dir)) {
  dir.create(tif_dir, recursive = TRUE)
}

# Load the harmonized continental data
continental_puvsp <- read.csv(here::here("Data", "My dataframes", "continental_puvsp_harmonised.csv"))
continental_coords <- read.csv(here::here("Data", "My dataframes", "continental_pu_coords.csv"))

# Join coordinates with PUVSP data
puvsp_with_coords <- continental_puvsp %>%
  left_join(continental_coords, by = c("id" = "id"))

# Get unique species IDs
species_ids <- unique(continental_puvsp$sp)

# Find the extent of your study area
x_range <- range(continental_coords$Coord_x, na.rm = TRUE)
y_range <- range(continental_coords$Coord_y, na.rm = TRUE)

# Define resolution as 0.5 degrees as specified
res <- 0.5

# Create rasters for each species
for (sp_id in species_ids) {
  # Filter data for this species
  sp_data <- puvsp_with_coords %>% 
    filter(sp == sp_id)
  
  # Get species name from first matching record
  sp_name <- sp_data$species_name[1]
  
  # Create safe filename (remove spaces, special characters)
  safe_name <- gsub("[^a-zA-Z0-9]", "_", sp_name)
  
  # Skip if no data for this species
  if (nrow(sp_data) == 0) {
    cat("No data for species ID:", sp_id, "\n")
    next
  }
  
  # Create empty raster with the extent of our data
  r <- raster(
    xmn = x_range[1] - res/2,
    xmx = x_range[2] + res/2,
    ymn = y_range[1] - res/2,
    ymx = y_range[2] + res/2,
    resolution = res
  )
  
  # Use rasterize to convert point data to raster
  # The 'amount' column is used as the value (presence/probability)
  r_filled <- rasterize(
    x = sp_data[, c("Coord_x", "Coord_y")],
    y = r,
    field = sp_data$amount,
    fun = max  # If multiple points in same cell, take maximum value
  )
  
  # File name with species name only
  filename <- file.path(tif_dir, paste0(safe_name, ".tif"))
  
  # Write to file
  writeRaster(r_filled, filename, format = "GTiff", overwrite = TRUE)
  
  cat("Created raster for", sp_name, "\n")
}

cat("Process complete. Created TIF files in:", tif_dir, "\n")

# Check the tif files

library(raster)
library(ggplot2)
library(sf)
library(dplyr)
library(gridExtra)
library(here)

# Directory containing TIF files
tif_dir <- here::here("Data", "tif files continental")

# Get list of all TIF files
tif_files <- list.files(tif_dir, pattern = "\\.tif$", full.names = TRUE)

if (length(tif_files) == 0) {
  stop("No TIF files found in the directory!")
}

# Basic information about the TIF files
cat("Found", length(tif_files), "TIF files\n")
cat("First few files:\n")
for (i in 1:min(5, length(tif_files))) {
  cat("  -", basename(tif_files[i]), "\n")
}

# Function to create a simple visualization of a raster
plot_raster <- function(raster_file) {
  # Extract species name from filename
  filename <- basename(raster_file)
  species_name <- gsub("sp_\\d+_(.+)\\.tif", "\\1", filename)
  species_name <- gsub("_", " ", species_name)
  
  # Load raster
  r <- raster(raster_file)
  
  # Check for empty raster
  if (all(is.na(values(r)))) {
    cat("WARNING: Raster is empty for", filename, "\n")
    return(NULL)
  }
  
  # Convert to dataframe for ggplot
  r_df <- as.data.frame(r, xy = TRUE)
  names(r_df)[3] <- "value"
  
  # Count non-NA cells
  non_na_cells <- sum(!is.na(r_df$value))
  total_cells <- nrow(r_df)
  
  # Create plot
  p <- ggplot(r_df, aes(x = x, y = y)) +
    geom_raster(aes(fill = value)) +
    scale_fill_viridis_c(na.value = "transparent") +
    coord_fixed() +
    theme_minimal() +
    labs(
      title = species_name,
      subtitle = paste0("Non-empty cells: ", non_na_cells, "/", total_cells),
      x = "Longitude",
      y = "Latitude",
      fill = "Value"
    )
  
  return(p)
}

# Sample a few TIF files to verify (adjust the number as needed)
set.seed(123)  # For reproducibility
sample_size <- min(9, length(tif_files))
sample_files <- sample(tif_files, sample_size)

# Create plots
plots <- lapply(sample_files, plot_raster)
plots <- plots[!sapply(plots, is.null)]  # Remove any NULL plots

# Arrange plots in a grid
if (length(plots) > 0) {
  grid_arranged <- do.call(gridExtra::grid.arrange, c(plots, ncol = 3))
  print(grid_arranged)
} else {
  cat("No valid plots to display!\n")
}

# More detailed verification
verification_results <- data.frame(
  filename = character(),
  non_empty_cells = numeric(),
  min_value = numeric(),
  max_value = numeric(),
  mean_value = numeric(),
  stringsAsFactors = FALSE
)

# Check each TIF file
for (tif_file in tif_files) {
  r <- raster(tif_file)
  values <- values(r)
  values_no_na <- values[!is.na(values)]
  
  # Skip if all NA
  if (length(values_no_na) == 0) {
    verification_results <- rbind(verification_results, data.frame(
      filename = basename(tif_file),
      non_empty_cells = 0,
      min_value = NA,
      max_value = NA,
      mean_value = NA,
      stringsAsFactors = FALSE
    ))
    next
  }
  
  verification_results <- rbind(verification_results, data.frame(
    filename = basename(tif_file),
    non_empty_cells = length(values_no_na),
    min_value = min(values_no_na),
    max_value = max(values_no_na),
    mean_value = mean(values_no_na),
    stringsAsFactors = FALSE
  ))
}

# Summary of verification
cat("\n==== Verification Summary ====\n")
cat("Total TIF files:", nrow(verification_results), "\n")
cat("Files with content:", sum(verification_results$non_empty_cells > 0), "\n")
cat("Empty files:", sum(verification_results$non_empty_cells == 0), "\n")

# Show files with potential issues (empty or very few cells)
problem_files <- verification_results %>%
  filter(non_empty_cells <= 5) %>%
  arrange(non_empty_cells)

if (nrow(problem_files) > 0) {
  cat("\nFiles with few or no data points:\n")
  print(problem_files)
}

# Save the verification summary
write.csv(
  verification_results,
  here::here("Data", "tif_verification_summary.csv"),
  row.names = FALSE
)

cat("\nVerification complete. Summary saved to 'Data/tif_verification_summary.csv'\n")

# Size of the rasters

library(raster)
library(dplyr)
library(here)

# Directory containing TIF files
tif_dir <- here::here("Data", "tif files continental")

# Get list of all TIF files
tif_files <- list.files(tif_dir, pattern = "\\.tif$", full.names = TRUE)

if (length(tif_files) == 0) {
  stop("No TIF files found in the directory!")
}

# Function to get raster dimensions
analyze_raster_size <- function(raster_file) {
  r <- raster(raster_file)
  
  # Extract dimensions
  dimensions <- dim(r)
  n_rows <- dimensions[1]
  n_cols <- dimensions[2]
  total_cells <- ncell(r)
  resolution <- res(r)
  extent <- extent(r)
  
  # Count non-NA cells
  values <- values(r)
  non_na_cells <- sum(!is.na(values))
  
  return(data.frame(
    filename = basename(raster_file),
    rows = n_rows,
    columns = n_cols,
    total_cells = total_cells,
    non_empty_cells = non_na_cells,
    empty_cells = total_cells - non_na_cells,
    percent_filled = round(100 * non_na_cells / total_cells, 2),
    x_resolution = resolution[1],
    y_resolution = resolution[2],
    x_min = extent@xmin,
    x_max = extent@xmax,
    y_min = extent@ymin,
    y_max = extent@ymax,
    stringsAsFactors = FALSE
  ))
}

# Analyze all rasters
raster_sizes <- do.call(rbind, lapply(tif_files, analyze_raster_size))

# Calculate some summary statistics
summary_stats <- data.frame(
  metric = c(
    "Number of rasters", 
    "Rows (min/mean/max)",
    "Columns (min/mean/max)",
    "Total cells (min/mean/max)",
    "X resolution",
    "Y resolution",
    "Average % of cells filled"
  ),
  value = c(
    nrow(raster_sizes),
    paste0(min(raster_sizes$rows), "/", round(mean(raster_sizes$rows), 1), "/", max(raster_sizes$rows)),
    paste0(min(raster_sizes$columns), "/", round(mean(raster_sizes$columns), 1), "/", max(raster_sizes$columns)),
    paste0(min(raster_sizes$total_cells), "/", round(mean(raster_sizes$total_cells), 1), "/", max(raster_sizes$total_cells)),
    unique(raster_sizes$x_resolution)[1],
    unique(raster_sizes$y_resolution)[1],
    round(mean(raster_sizes$percent_filled), 2)
  )
)

# Print summary
cat("\n===== Raster Size Analysis =====\n")
print(summary_stats, row.names = FALSE)

cat("\nSample of Individual Raster Analysis:\n")
print(head(raster_sizes %>% select(filename, rows, columns, total_cells, non_empty_cells, percent_filled)), row.names = FALSE)

# Verify all rasters have the same dimensions
uniform_dimensions <- length(unique(raster_sizes$rows)) == 1 && 
  length(unique(raster_sizes$columns)) == 1

cat("\nAll rasters have the same dimensions:", uniform_dimensions, "\n")
if (!uniform_dimensions) {
  cat("WARNING: Rasters have different dimensions. This may cause issues for some analyses.\n")
  
  # Show the different dimensions
  unique_dims <- raster_sizes %>%
    select(rows, columns, total_cells) %>%
    distinct() %>%
    arrange(rows, columns)
  
  cat("Unique dimensions (rows x columns):\n")
  print(unique_dims, row.names = FALSE)
}

# Verify all rasters have the same extent
uniform_extent <- length(unique(raster_sizes$x_min)) == 1 && 
  length(unique(raster_sizes$x_max)) == 1 &&
  length(unique(raster_sizes$y_min)) == 1 &&
  length(unique(raster_sizes$y_max)) == 1

cat("\nAll rasters have the same extent:", uniform_extent, "\n")

# Save the detailed analysis
write.csv(
  raster_sizes,
  here::here("Data", "tif_size_analysis.csv"),
  row.names = FALSE
)

cat("\nAnalysis complete. Detailed results saved to 'Data/tif_size_analysis.csv'\n")

# Create .csv file for new CAPTAIN model (continental) ----

metrics_10_iucn_continental <- readRDS(here::here("Data", "My dataframes", 
                                      "continental_shark_conservation_metrics_10_harmonised_IUCN.rds"))
metrics_30_iucn_continental <- readRDS(here::here("Data", "My dataframes", 
                                      "continental_shark_conservation_metrics_30_harmonised_IUCN.rds"))

# Build the species x metric table by matching species IDs (not row positions).
# Fix (November 2026): the previous version used cbind.data.frame() on
# iucn_GE$info$species_name (lookup order) and transformed$amount (species ID order),
# which attached the IUCN, FUSE and EDGE2 values to the wrong species names.
global_species_lookup <- readRDS(here::here("Data", "My dataframes", "global_species_lookup.rds"))

build_metrics_table <- function(metrics) {
  get_metric <- function(metric, name) {
    data.frame(sp = metrics[[metric]]$transformed$sp,
               value = as.numeric(metrics[[metric]]$transformed$amount)) %>%
      dplyr::rename(!!name := value)
  }
  get_metric("iucn_GE", "IUCN") %>%
    dplyr::left_join(get_metric("FUSE", "FUSE"), by = "sp") %>%
    dplyr::left_join(get_metric("EDGE2", "EDGE2"), by = "sp") %>%
    dplyr::left_join(global_species_lookup, by = c("sp" = "species_id")) %>%
    dplyr::transmute(`Species name` = species_name, IUCN, FUSE, EDGE2) %>%
    dplyr::arrange(`Species name`)
}

metrics_10_iucn_continental_csv <- build_metrics_table(metrics_10_iucn_continental)

metrics_30_iucn_continental_csv <- build_metrics_table(metrics_30_iucn_continental)

# Save the 10% metrics dataframe
write.csv(metrics_10_iucn_continental_csv, 
          file = here::here("Data", "My dataframes","continental_shark_conservation_metrics_10_harmonised_IUCN_csv.csv"), 
          row.names = FALSE)

# Save the 30% metrics dataframe
write.csv(metrics_30_iucn_continental_csv, 
          file = here::here("Data", "My dataframes","continental_shark_conservation_metrics_30_harmonised_IUCN_csv.csv"), 
          row.names = FALSE)

# Categorise these three values 

# Create a new dataframe for the categorized data
metrics_10_categorized <- metrics_10_iucn_continental_csv

# For IUCN, map the existing unique values to categories 1-5
# Based on the unique values: 0.025, 0.250, 0.500, 0.750, 1.000
metrics_10_categorized$IUCN <- factor(metrics_10_categorized$IUCN,
                                          levels = c(0.025, 0.250, 0.500, 0.750, 1.000),
                                          labels = c(1, 2, 3, 4, 5))

# For FUSE, categorize into 5 classes
metrics_10_categorized$FUSE <- cut(metrics_10_categorized$FUSE, 
                                       breaks = c(-Inf, 0.2, 0.4, 0.6, 0.8, Inf),
                                       labels = c(1, 2, 3, 4, 5))

# For EDGE2, categorize into 5 classes
metrics_10_categorized$EDGE2 <- cut(metrics_10_categorized$EDGE2, 
                                        breaks = c(-Inf, 0.2, 0.4, 0.6, 0.8, Inf),
                                        labels = c(1, 2, 3, 4, 5))

# If you want to update the original column instead of creating a new one:
metrics_10_categorized$`Species name` <- gsub(" ", "_", metrics_10_categorized$`Species name`)

# View the updated dataframe
head(metrics_10_categorized)

# Save the 10% metrics dataframe
write.csv(metrics_10_categorized, 
          file = here::here("Data", "My dataframes","continental_shark_conservation_metrics_10_harmonised_IUCN_categories.csv"), 
          row.names = FALSE)

#Compare categories
# First, ensure we're just working with the categorized 10% dataset
cat_data <- metrics_10_categorized

# Create a summary table showing the distribution of species across categories for each metric
summary_table <- data.frame(
  Metric = c(rep("IUCN", 5), rep("FUSE", 5), rep("EDGE2", 5)),
  Category = rep(1:5, 3),
  Count = c(
    table(cat_data$IUCN_cat),
    table(cat_data$FUSE_cat),
    table(cat_data$EDGE2_cat)
  )
)

# Cross-tabulation between pairs of metrics
iucn_vs_fuse <- table(cat_data$IUCN_cat, cat_data$FUSE_cat)
iucn_vs_edge2 <- table(cat_data$IUCN_cat, cat_data$EDGE2_cat)
fuse_vs_edge2 <- table(cat_data$FUSE_cat, cat_data$EDGE2_cat)

# Calculate agreement/disagreement
# Count how many species have the same category across all three metrics
full_agreement <- sum(cat_data$IUCN_cat == cat_data$FUSE_cat & 
                        cat_data$IUCN_cat == cat_data$EDGE2_cat)

# Count how many species have different categories for each metric
all_different <- sum(cat_data$IUCN_cat != cat_data$FUSE_cat & 
                       cat_data$IUCN_cat != cat_data$EDGE2_cat &
                       cat_data$FUSE_cat != cat_data$EDGE2_cat)

# Calculate pairwise agreement percentages
iucn_fuse_agreement <- sum(cat_data$IUCN_cat == cat_data$FUSE_cat) / nrow(cat_data) * 100
iucn_edge2_agreement <- sum(cat_data$IUCN_cat == cat_data$EDGE2_cat) / nrow(cat_data) * 100
fuse_edge2_agreement <- sum(cat_data$FUSE_cat == cat_data$EDGE2_cat) / nrow(cat_data) * 100

# Identify species with the biggest disagreements (difference of 2+ categories)
cat_data$max_diff <- apply(cat_data[,c("IUCN_cat", "FUSE_cat", "EDGE2_cat")], 1, function(x) {
  x <- as.numeric(as.character(x))
  max(x) - min(x)
})

large_disagreement <- cat_data[cat_data$max_diff >= 2, c("Species name", "IUCN_cat", "FUSE_cat", "EDGE2_cat")]

# Check which metric tends to give higher/lower categories
cat_data$IUCN_numeric <- as.numeric(as.character(cat_data$IUCN_cat))
cat_data$FUSE_numeric <- as.numeric(as.character(cat_data$FUSE_cat))
cat_data$EDGE2_numeric <- as.numeric(as.character(cat_data$EDGE2_cat))

avg_categories <- c(
  mean(cat_data$IUCN_numeric, na.rm=TRUE),
  mean(cat_data$FUSE_numeric, na.rm=TRUE),
  mean(cat_data$EDGE2_numeric, na.rm=TRUE)
)
names(avg_categories) <- c("IUCN", "FUSE", "EDGE2")

# Print results
print(summary_table)
cat("\nCross-tabulation of IUCN vs FUSE:\n")
print(iucn_vs_fuse)
cat("\nCross-tabulation of IUCN vs EDGE2:\n")
print(iucn_vs_edge2)
cat("\nCross-tabulation of FUSE vs EDGE2:\n")
print(fuse_vs_edge2)

cat("\nPercentage of species with full agreement across all metrics: ", 
    round(full_agreement/nrow(cat_data)*100, 1), "%\n")
cat("Percentage of species with completely different categories: ", 
    round(all_different/nrow(cat_data)*100, 1), "%\n")

cat("\nPairwise agreement percentages:\n")
cat("IUCN vs FUSE: ", round(iucn_fuse_agreement, 1), "%\n")
cat("IUCN vs EDGE2: ", round(iucn_edge2_agreement, 1), "%\n")
cat("FUSE vs EDGE2: ", round(fuse_edge2_agreement, 1), "%\n")

cat("\nSpecies with large disagreements (difference of 2+ categories):\n")
print(large_disagreement)

cat("\nAverage category by metric:\n")
print(round(avg_categories, 2))

# Create individual .tif files (high seas) ----
library(raster)
library(dplyr)
library(tidyr)
library(here)

# Create directory if it doesn't exist
tif_dir <- here::here("Data", "tif files highseas")
if (!dir.exists(tif_dir)) {
  dir.create(tif_dir, recursive = TRUE)
}

# Load the harmonized high seas data
highseas_puvsp <- read.csv(here::here("Data", "My dataframes", "highseas_puvsp_harmonised.csv"))
highseas_coords <- read.csv(here::here("Data", "My dataframes", "highseas_pu_coords.csv"))

# Join coordinates with PUVSP data
puvsp_with_coords <- highseas_puvsp %>%
  left_join(highseas_coords, by = c("id" = "id"))

# Get unique species IDs
species_ids <- unique(highseas_puvsp$sp)

# Find the extent of your study area
x_range <- range(highseas_coords$Coord_x, na.rm = TRUE)
y_range <- range(highseas_coords$Coord_y, na.rm = TRUE)

# Define resolution as 0.5 degrees as specified
res <- 0.5

# Create rasters for each species
for (sp_id in species_ids) {
  # Filter data for this species
  sp_data <- puvsp_with_coords %>% 
    filter(sp == sp_id)
  
  # Get species name from first matching record
  sp_name <- sp_data$species_name[1]
  
  # Create safe filename (remove spaces, special characters)
  safe_name <- gsub("[^a-zA-Z0-9]", "_", sp_name)
  
  # Skip if no data for this species
  if (nrow(sp_data) == 0) {
    cat("No data for species ID:", sp_id, "\n")
    next
  }
  
  # Create empty raster with the extent of our data
  r <- raster(
    xmn = x_range[1] - res/2,
    xmx = x_range[2] + res/2,
    ymn = y_range[1] - res/2,
    ymx = y_range[2] + res/2,
    resolution = res
  )
  
  # Use rasterize to convert point data to raster
  # The 'amount' column is used as the value (presence/probability)
  r_filled <- rasterize(
    x = sp_data[, c("Coord_x", "Coord_y")],
    y = r,
    field = sp_data$amount,
    fun = max  # If multiple points in same cell, take maximum value
  )
  
  # File name with species name only
  filename <- file.path(tif_dir, paste0(safe_name, ".tif"))
  
  # Write to file
  writeRaster(r_filled, filename, format = "GTiff", overwrite = TRUE)
  
  cat("Created raster for", sp_name, "\n")
}

cat("Process complete. Created TIF files in:", tif_dir, "\n")

# Check the tif files

library(raster)
library(ggplot2)
library(sf)
library(dplyr)
library(gridExtra)
library(here)

# Directory containing TIF files
tif_dir <- here::here("Data", "tif files highseas")

# Get list of all TIF files
tif_files <- list.files(tif_dir, pattern = "\\.tif$", full.names = TRUE)

if (length(tif_files) == 0) {
  stop("No TIF files found in the directory!")
}

# Basic information about the TIF files
cat("Found", length(tif_files), "TIF files\n")
cat("First few files:\n")
for (i in 1:min(5, length(tif_files))) {
  cat("  -", basename(tif_files[i]), "\n")
}

# Function to create a simple visualization of a raster
plot_raster <- function(raster_file) {
  # Extract species name from filename
  filename <- basename(raster_file)
  species_name <- gsub("sp_\\d+_(.+)\\.tif", "\\1", filename)
  species_name <- gsub("_", " ", species_name)
  
  # Load raster
  r <- raster(raster_file)
  
  # Check for empty raster
  if (all(is.na(values(r)))) {
    cat("WARNING: Raster is empty for", filename, "\n")
    return(NULL)
  }
  
  # Convert to dataframe for ggplot
  r_df <- as.data.frame(r, xy = TRUE)
  names(r_df)[3] <- "value"
  
  # Count non-NA cells
  non_na_cells <- sum(!is.na(r_df$value))
  total_cells <- nrow(r_df)
  
  # Create plot
  p <- ggplot(r_df, aes(x = x, y = y)) +
    geom_raster(aes(fill = value)) +
    scale_fill_viridis_c(na.value = "transparent") +
    coord_fixed() +
    theme_minimal() +
    labs(
      title = species_name,
      subtitle = paste0("Non-empty cells: ", non_na_cells, "/", total_cells),
      x = "Longitude",
      y = "Latitude",
      fill = "Value"
    )
  
  return(p)
}

# Sample a few TIF files to verify (adjust the number as needed)
set.seed(123)  # For reproducibility
sample_size <- min(9, length(tif_files))
sample_files <- sample(tif_files, sample_size)

# Create plots
plots <- lapply(sample_files, plot_raster)
plots <- plots[!sapply(plots, is.null)]  # Remove any NULL plots

# Arrange plots in a grid
if (length(plots) > 0) {
  grid_arranged <- do.call(gridExtra::grid.arrange, c(plots, ncol = 3))
  print(grid_arranged)
} else {
  cat("No valid plots to display!\n")
}

# More detailed verification
verification_results <- data.frame(
  filename = character(),
  non_empty_cells = numeric(),
  min_value = numeric(),
  max_value = numeric(),
  mean_value = numeric(),
  stringsAsFactors = FALSE
)

# Check each TIF file
for (tif_file in tif_files) {
  r <- raster(tif_file)
  values <- values(r)
  values_no_na <- values[!is.na(values)]
  
  # Skip if all NA
  if (length(values_no_na) == 0) {
    verification_results <- rbind(verification_results, data.frame(
      filename = basename(tif_file),
      non_empty_cells = 0,
      min_value = NA,
      max_value = NA,
      mean_value = NA,
      stringsAsFactors = FALSE
    ))
    next
  }
  
  verification_results <- rbind(verification_results, data.frame(
    filename = basename(tif_file),
    non_empty_cells = length(values_no_na),
    min_value = min(values_no_na),
    max_value = max(values_no_na),
    mean_value = mean(values_no_na),
    stringsAsFactors = FALSE
  ))
}

# Summary of verification
cat("\n==== Verification Summary ====\n")
cat("Total TIF files:", nrow(verification_results), "\n")
cat("Files with content:", sum(verification_results$non_empty_cells > 0), "\n")
cat("Empty files:", sum(verification_results$non_empty_cells == 0), "\n")

# Show files with potential issues (empty or very few cells)
problem_files <- verification_results %>%
  filter(non_empty_cells <= 5) %>%
  arrange(non_empty_cells)

if (nrow(problem_files) > 0) {
  cat("\nFiles with few or no data points:\n")
  print(problem_files)
}

# Save the verification summary
write.csv(
  verification_results,
  here::here("Data", "tif_verification_summary_highseas.csv"),
  row.names = FALSE
)

cat("\nVerification complete. Summary saved to 'Data/tif_verification_summary_highseas.csv'\n")

# Size of the rasters

library(raster)
library(dplyr)
library(here)

# Directory containing TIF files
tif_dir <- here::here("Data", "tif files highseas")

# Get list of all TIF files
tif_files <- list.files(tif_dir, pattern = "\\.tif$", full.names = TRUE)

if (length(tif_files) == 0) {
  stop("No TIF files found in the directory!")
}

# Function to get raster dimensions
analyze_raster_size <- function(raster_file) {
  r <- raster(raster_file)
  
  # Extract dimensions
  dimensions <- dim(r)
  n_rows <- dimensions[1]
  n_cols <- dimensions[2]
  total_cells <- ncell(r)
  resolution <- res(r)
  extent <- extent(r)
  
  # Count non-NA cells
  values <- values(r)
  non_na_cells <- sum(!is.na(values))
  
  return(data.frame(
    filename = basename(raster_file),
    rows = n_rows,
    columns = n_cols,
    total_cells = total_cells,
    non_empty_cells = non_na_cells,
    empty_cells = total_cells - non_na_cells,
    percent_filled = round(100 * non_na_cells / total_cells, 2),
    x_resolution = resolution[1],
    y_resolution = resolution[2],
    x_min = extent@xmin,
    x_max = extent@xmax,
    y_min = extent@ymin,
    y_max = extent@ymax,
    stringsAsFactors = FALSE
  ))
}

# Analyze all rasters
raster_sizes <- do.call(rbind, lapply(tif_files, analyze_raster_size))

# Calculate some summary statistics
summary_stats <- data.frame(
  metric = c(
    "Number of rasters", 
    "Rows (min/mean/max)",
    "Columns (min/mean/max)",
    "Total cells (min/mean/max)",
    "X resolution",
    "Y resolution",
    "Average % of cells filled"
  ),
  value = c(
    nrow(raster_sizes),
    paste0(min(raster_sizes$rows), "/", round(mean(raster_sizes$rows), 1), "/", max(raster_sizes$rows)),
    paste0(min(raster_sizes$columns), "/", round(mean(raster_sizes$columns), 1), "/", max(raster_sizes$columns)),
    paste0(min(raster_sizes$total_cells), "/", round(mean(raster_sizes$total_cells), 1), "/", max(raster_sizes$total_cells)),
    unique(raster_sizes$x_resolution)[1],
    unique(raster_sizes$y_resolution)[1],
    round(mean(raster_sizes$percent_filled), 2)
  )
)

# Print summary
cat("\n===== Raster Size Analysis =====\n")
print(summary_stats, row.names = FALSE)

cat("\nSample of Individual Raster Analysis:\n")
print(head(raster_sizes %>% select(filename, rows, columns, total_cells, non_empty_cells, percent_filled)), row.names = FALSE)

# Verify all rasters have the same dimensions
uniform_dimensions <- length(unique(raster_sizes$rows)) == 1 && 
  length(unique(raster_sizes$columns)) == 1

cat("\nAll rasters have the same dimensions:", uniform_dimensions, "\n")
if (!uniform_dimensions) {
  cat("WARNING: Rasters have different dimensions. This may cause issues for some analyses.\n")
  
  # Show the different dimensions
  unique_dims <- raster_sizes %>%
    select(rows, columns, total_cells) %>%
    distinct() %>%
    arrange(rows, columns)
  
  cat("Unique dimensions (rows x columns):\n")
  print(unique_dims, row.names = FALSE)
}

# Verify all rasters have the same extent
uniform_extent <- length(unique(raster_sizes$x_min)) == 1 && 
  length(unique(raster_sizes$x_max)) == 1 &&
  length(unique(raster_sizes$y_min)) == 1 &&
  length(unique(raster_sizes$y_max)) == 1

cat("\nAll rasters have the same extent:", uniform_extent, "\n")

# Save the detailed analysis
write.csv(
  raster_sizes,
  here::here("Data", "tif_size_analysis_highseas.csv"),
  row.names = FALSE
)

cat("\nAnalysis complete. Detailed results saved to 'Data/tif_size_analysis_highseas.csv'\n")
# Create .csv file for new CAPTAIN model (high seas) ----

metrics_10_iucn_highseas <- readRDS(here::here("Data", "My dataframes", 
                                               "highseas_shark_conservation_metrics_10_harmonised_IUCN.rds"))
metrics_30_iucn_highseas <- readRDS(here::here("Data", "My dataframes", 
                                               "highseas_shark_conservation_metrics_30_harmonised_IUCN.rds"))

metrics_10_iucn_highseas_csv <- build_metrics_table(metrics_10_iucn_highseas)

metrics_30_iucn_highseas_csv <- build_metrics_table(metrics_30_iucn_highseas)

# Save the 10% metrics dataframe
write.csv(metrics_10_iucn_highseas_csv, 
          file = here::here("Data", "My dataframes","highseas_shark_conservation_metrics_10_harmonised_IUCN_csv.csv"), 
          row.names = FALSE)

# Save the 30% metrics dataframe
write.csv(metrics_30_iucn_highseas_csv, 
          file = here::here("Data", "My dataframes","highseas_shark_conservation_metrics_30_harmonised_IUCN_csv.csv"), 
          row.names = FALSE)

# Categorise these three values 

# Create a new dataframe for the categorized data
metrics_10_categorized <- metrics_10_iucn_highseas_csv

# For IUCN, map the existing unique values to categories 1-5
# Based on the unique values: 0.025, 0.250, 0.500, 0.750, 1.000
metrics_10_categorized$IUCN <- factor(metrics_10_categorized$IUCN,
                                      levels = c(0.025, 0.250, 0.500, 0.750, 1.000),
                                      labels = c(1, 2, 3, 4, 5))

# For FUSE, categorize into 5 classes
metrics_10_categorized$FUSE <- cut(metrics_10_categorized$FUSE, 
                                   breaks = c(-Inf, 0.2, 0.4, 0.6, 0.8, Inf),
                                   labels = c(1, 2, 3, 4, 5))

# For EDGE2, categorize into 5 classes
metrics_10_categorized$EDGE2 <- cut(metrics_10_categorized$EDGE2, 
                                    breaks = c(-Inf, 0.2, 0.4, 0.6, 0.8, Inf),
                                    labels = c(1, 2, 3, 4, 5))

# If you want to update the original column instead of creating a new one:
metrics_10_categorized$`Species name` <- gsub(" ", "_", metrics_10_categorized$`Species name`)

# View the updated dataframe
head(metrics_10_categorized)

# Save the 10% metrics dataframe
write.csv(metrics_10_categorized, 
          file = here::here("Data", "My dataframes","highseas_shark_conservation_metrics_10_harmonised_IUCN_categories.csv"), 
          row.names = FALSE)

#Compare categories
# First, ensure we're just working with the categorized 10% dataset
cat_data <- metrics_10_categorized

# Create a summary table showing the distribution of species across categories for each metric
summary_table <- data.frame(
  Metric = c(rep("IUCN", 5), rep("FUSE", 5), rep("EDGE2", 5)),
  Category = rep(1:5, 3),
  Count = c(
    table(cat_data$IUCN),
    table(cat_data$FUSE),
    table(cat_data$EDGE2)
  )
)

# Cross-tabulation between pairs of metrics
iucn_vs_fuse <- table(cat_data$IUCN, cat_data$FUSE)
iucn_vs_edge2 <- table(cat_data$IUCN, cat_data$EDGE2)
fuse_vs_edge2 <- table(cat_data$FUSE, cat_data$EDGE2)

# Calculate agreement/disagreement
# Count how many species have the same category across all three metrics
full_agreement <- sum(cat_data$IUCN == cat_data$FUSE & 
                        cat_data$IUCN == cat_data$EDGE2)

# Count how many species have different categories for each metric
all_different <- sum(cat_data$IUCN != cat_data$FUSE & 
                       cat_data$IUCN != cat_data$EDGE2 &
                       cat_data$FUSE != cat_data$EDGE2)

# Calculate pairwise agreement percentages
iucn_fuse_agreement <- sum(cat_data$IUCN == cat_data$FUSE) / nrow(cat_data) * 100
iucn_edge2_agreement <- sum(cat_data$IUCN == cat_data$EDGE2) / nrow(cat_data) * 100
fuse_edge2_agreement <- sum(cat_data$FUSE == cat_data$EDGE2) / nrow(cat_data) * 100

# Identify species with the biggest disagreements (difference of 2+ categories)
cat_data$max_diff <- apply(cat_data[,c("IUCN", "FUSE", "EDGE2")], 1, function(x) {
  x <- as.numeric(as.character(x))
  max(x) - min(x)
})

large_disagreement <- cat_data[cat_data$max_diff >= 2, c("Species name", "IUCN", "FUSE", "EDGE2")]

# Check which metric tends to give higher/lower categories
cat_data$IUCN_numeric <- as.numeric(as.character(cat_data$IUCN))
cat_data$FUSE_numeric <- as.numeric(as.character(cat_data$FUSE))
cat_data$EDGE2_numeric <- as.numeric(as.character(cat_data$EDGE2))

avg_categories <- c(
  mean(cat_data$IUCN_numeric, na.rm=TRUE),
  mean(cat_data$FUSE_numeric, na.rm=TRUE),
  mean(cat_data$EDGE2_numeric, na.rm=TRUE)
)
names(avg_categories) <- c("IUCN", "FUSE", "EDGE2")

# Print results
print(summary_table)
cat("\nCross-tabulation of IUCN vs FUSE:\n")
print(iucn_vs_fuse)
cat("\nCross-tabulation of IUCN vs EDGE2:\n")
print(iucn_vs_edge2)
cat("\nCross-tabulation of FUSE vs EDGE2:\n")
print(fuse_vs_edge2)

cat("\nPercentage of species with full agreement across all metrics: ", 
    round(full_agreement/nrow(cat_data)*100, 1), "%\n")
cat("Percentage of species with completely different categories: ", 
    round(all_different/nrow(cat_data)*100, 1), "%\n")

cat("\nPairwise agreement percentages:\n")
cat("IUCN vs FUSE: ", round(iucn_fuse_agreement, 1), "%\n")
cat("IUCN vs EDGE2: ", round(iucn_edge2_agreement, 1), "%\n")
cat("FUSE vs EDGE2: ", round(fuse_edge2_agreement, 1), "%\n")

cat("\nSpecies with large disagreements (difference of 2+ categories):\n")
print(large_disagreement)

cat("\nAverage category by metric:\n")
print(round(avg_categories, 2))
