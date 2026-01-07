#!/usr/bin/env python3
"""
Script to remove exact duplicate images from dataset by deleting duplicates in place.
"""

import os
from pathlib import Path
from typing import Sequence


def remove_duplicates(duplicate_sets: Sequence[Sequence[str]], work_folder: str | Path) -> None:
    """
    Removes duplicate images from the work folder directly.
    Always keeps the first image in each duplicate set and deletes the rest.
    
    Args:
        duplicate_sets: Sequence of sequences containing duplicate image paths
        work_folder: Path to the folder containing images (duplicates will be deleted from here)
    
    Returns:
        Tuple containing (number of sets processed, number of images deleted)
    """
    # Convert to Path object and resolve to absolute path
    work_folder = Path(work_folder).expanduser().resolve()
    
    print(f"Work folder: {work_folder}")
    
    # Get number of duplicate sets
    num_sets = len(duplicate_sets)
    
    print(f"Found {num_sets} sets of exact duplicates")
    
    # Counter for deleted images
    deleted_count = 0
    
    # Process each set of duplicates and delete all but the first
    for i, duplicate_set in enumerate(duplicate_sets):
        if not duplicate_set:  # Skip empty sets
            continue
            
        print(f"Processing duplicate set {i+1}/{num_sets} with {len(duplicate_set)} images")
        
        # Keep the first image, delete the rest
        for image_path in duplicate_set[1:]:
            # Construct full path to the duplicate file
            full_path = work_folder / image_path
            
            try:
                if full_path.exists():
                    os.remove(full_path)
                    print(f"Deleted: {image_path}")
                    deleted_count += 1
                else:
                    print(f"Warning: File not found: {image_path}")
            except Exception as e:
                print(f"Error deleting {image_path}: {e}")
    
    print(f"\nSummary: Processed {num_sets} duplicate sets")
    print(f"Deleted {deleted_count} duplicate images")
