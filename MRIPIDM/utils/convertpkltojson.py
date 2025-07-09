import pickle
import json
import os 

root_path = r"D:/Projects/MRIPIDMoutput/labeledSpace"

for folder in os.listdir(root_path):
    print(folder)
    with open(f"{root_path}/{folder}", "rb") as f:
        data = pickle.load(f)

    dest_path = r"D:/Projects/MRIPIDMoutput/labeledSpaceJSON"
    os.makedirs(dest_path, exist_ok=True)

    with open(f"{dest_path}/{folder}.json", "w") as f: #names are going to appear as .pkl.json
        json.dump(data, f)