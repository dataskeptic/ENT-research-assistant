import os
from pathlib import Path

def count_papers(data_dir: str = "data/parsed"):
    path = Path(data_dir)
    if not path.exists():
        print(f"Directory '{data_dir}' does not exist.")
        return
    
    count = sum(1 for item in path.iterdir() if item.is_file() and item.name.endswith('.json'))
    print(f"There are {count} parsed papers in '{data_dir}'")

if __name__ == "__main__":
    count_papers()
