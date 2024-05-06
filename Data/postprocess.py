import os

### Rename files in a directory

# Get current working directory
target_folder = "p30"
directory = os.getcwd() + "\\" + target_folder

# Specify the old and new prefixes
old_prefix = "P30"
new_prefix = "p30"

# Iterate over all files in the specified directory

for filename in os.listdir(directory):
    # Check if the file starts with the old prefix
    if filename.startswith(old_prefix):
        # Form the new filename by replacing the old prefix with the new one
        new_filename = filename.replace(old_prefix, new_prefix)

        # Form the full old and new file paths
        old_filepath = os.path.join(directory, filename)
        new_filepath = os.path.join(directory, new_filename)

        # Rename the file
        os.rename(old_filepath, new_filepath)
        print(f"Renamed '{old_filepath}' to '{new_filepath}'")

