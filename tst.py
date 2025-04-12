import os

# Relative path to the text file
file_path = os.path.join("arshan.txt")

# Open and read the file
with open(file_path, "r") as file:
    content = file.read()

print("File content:\n", content)
