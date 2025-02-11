# This script is used to upload the data to the Hugging Face Hub

import os
import csv
from datasets import load_dataset
from tqdm import tqdm
# Create metadata.csv in ./op directory
with open('./op/train/metadata.csv', 'w', newline='') as csvfile:
    writer = csv.writer(csvfile)
    writer.writerow(['file_name', 'actions'])
    
    # Iterate through all txt files in ./op
    for root, dirs, files in tqdm(os.walk('./op')):
        dirname = os.path.basename(root)
        if dirname.startswith('video_'):
            index = dirname.split('_')[1]
            index = str(int(index)-1).zfill(3)
            textfile = os.path.join(root, 'video_' + index + '.txt')
            if os.path.exists(textfile):
                with open(textfile, 'r') as f:
                    actions = f.readlines()
                    actions = [action.strip() for action in actions]
                for filename in files:
                    if filename.endswith('.jpg'):
                        index = filename.split('_')[1].split('.')[0].strip('0')
                        writer.writerow([dirname + '/' + filename, actions[int(index) - 1]])

# Load the dataset with the metadata file
dataset = load_dataset("imagefolder", 
                      data_dir="./op")

print(dataset['train'][0])

print(dataset)