#!/bin/bash

# Base directory
base_dir="data_collection/recordings"

# Files to copy
files=("panas.csv" "readme.txt" "sssq.csv" "stai.csv" "timings.csv")

# Loop through S1 to S20
for i in {1..20}
do
  # Create directory S1, S2, ..., S20
  dir="$base_dir/S$i"
  mkdir -p "$dir"
  
  # Copy files into the directory
  for file in "${files[@]}"
  do
    cp "$base_dir/$file" "$dir/"
  done
done
