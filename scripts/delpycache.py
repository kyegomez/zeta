import os
import shutil
import sys


def delete_pycache(directory):
    for root, dirs, files in os.walk(directory):
        if "__pycache__" in dirs:
            shutil.rmtree(os.path.join(root, "__pycache__"))


if __name__ == "__main__":
    if len(sys.argv) != 2:
        print("Usage: python delete_pycache.py <directory>")
        sys.exit(1)

    directory = sys.argv[1]
    delete_pycache(directory)
    print(f"__pycache__ directories deleted in {directory}")
