import numpy as np
import pandas as pd
import os
import pickle
import argparse

parser = argparse.ArgumentParser(description='read pkl file')
parser.add_argument('--path', default="./results/finetune_100000_offline_s1_t0.33_k6_p1_5_samples.pkl",
                    type=str, help="pkl file path")
args = parser.parse_args()

with open(args.path, 'rb') as file:   #load the .pkl file
    data = pickle.load(file)

if not isinstance(data, dict):
    raise ValueError("The content of the pkl file should be a dictionary.")

save_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'emotion')
if not os.path.exists(save_dir):
    os.makedirs(save_dir)

for key,value in data.items():
    for i in range(value[0].shape):
        array_2d=value[i]
        keyname=key.replace('/','_')
        df = pd.DataFrame(array_2d)
        filename = os.path.join(save_dir, f'{keyname}_{i}.csv')
        df.to_csv(filename, index=False)



