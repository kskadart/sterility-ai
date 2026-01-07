#!/usr/bin/env python3
"""
Tests for the remove_duplicates.py script
"""
import shutil
import tempfile
import unittest
from pathlib import Path

# Use absolute imports as per user's request
from scr.remove_duplicates import remove_duplicates


class TestRemoveDuplicates(unittest.TestCase):
    def setUp(self):
        # Create temporary directory for testing
        self.temp_dir = tempfile.mkdtemp()
        self.work_folder = Path(self.temp_dir) / "test_dataset"
        self.work_folder.mkdir(exist_ok=True)
        
        # Create some fake image files
        self.create_fake_images()

    def tearDown(self):
        # Clean up after tests
        shutil.rmtree(self.temp_dir)

    def create_fake_images(self):
        # Create a few test files
        for i in range(5):
            img_path = self.work_folder / f"img_{i}.jpg"
            with open(img_path, "w") as f:
                f.write(f"Test content for image {i}")

    def test_remove_duplicates_always_keep_first(self):
        # Create duplicate sets directly
        duplicate_sets = [
            ["img_0.jpg", "img_1.jpg", "img_2.jpg"],  # First set with 3 duplicates
            ["img_3.jpg", "img_4.jpg"]                # Second set with 2 duplicates
        ]
        
        # Call the function (always keeps first image)
        num_sets, deleted_count = remove_duplicates(
            duplicate_sets,
            self.work_folder
        )
        
        # Verify results
        self.assertEqual(num_sets, 2)  # Two sets processed
        self.assertEqual(deleted_count, 3)  # 2 from first set + 1 from second set
        
        # Verify remaining files in the work folder
        remaining_files = list(self.work_folder.glob("*.jpg"))
        self.assertEqual(len(remaining_files), 2)  # Should have only img_0.jpg and img_3.jpg
        
        # Verify the right files were kept (first in each duplicate set)
        self.assertTrue((self.work_folder / "img_0.jpg").exists())
        self.assertTrue((self.work_folder / "img_3.jpg").exists())
        
        # Verify the duplicate files were deleted
        self.assertFalse((self.work_folder / "img_1.jpg").exists())
        self.assertFalse((self.work_folder / "img_2.jpg").exists())
        self.assertFalse((self.work_folder / "img_4.jpg").exists())
        
        # Verify no clean_dataset folder was created
        clean_dataset_path = self.work_folder / "clean_dataset"
        self.assertFalse(clean_dataset_path.exists())

    def test_remove_duplicates_empty_sets(self):
        # Test with empty duplicate sets and single image sets
        duplicate_sets = [
            [],  # Empty set
            ["img_0.jpg"],  # Single image (no duplicates to remove)
            ["img_1.jpg", "img_2.jpg"]  # Actual duplicates
        ]
        
        # Call the function
        num_sets, deleted_count = remove_duplicates(
            duplicate_sets,
            self.work_folder
        )
        
        # Verify results - only the last set with actual duplicates should contribute to deleted count
        self.assertEqual(num_sets, 3)  # Three sets processed
        self.assertEqual(deleted_count, 1)  # Only 1 image deleted (img_2.jpg)
        
        # Verify remaining files in the work folder
        remaining_files = list(self.work_folder.glob("*.jpg"))
        self.assertEqual(len(remaining_files), 4)  # Should have img_0.jpg, img_1.jpg, img_3.jpg, img_4.jpg
        
        # Verify the right files were kept
        self.assertTrue((self.work_folder / "img_0.jpg").exists())  # Single image, kept
        self.assertTrue((self.work_folder / "img_1.jpg").exists())  # First of duplicates, kept
        self.assertFalse((self.work_folder / "img_2.jpg").exists())  # Second of duplicates, deleted
        self.assertTrue((self.work_folder / "img_3.jpg").exists())  # Not in duplicate sets
        self.assertTrue((self.work_folder / "img_4.jpg").exists())  # Not in duplicate sets
        
        # Verify no clean_dataset folder was created
        clean_dataset_path = self.work_folder / "clean_dataset"
        self.assertFalse(clean_dataset_path.exists())


if __name__ == "__main__":
    unittest.main()
