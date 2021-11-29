import os,pickle
import numpy as np
import pandas as pd

root = './data/'
fps = {fp:root+fp for fp in os.listdir(root)}

demos = {}; count = 1; total = len(fps)

for i,dir_ in fps.items():
    
    print(f'{count}/{total}',end='\r');
    fp = dir_+'/demo.pickle'
    if os.path.exists(fp):
        with open(fp,'rb') as f:
            demos[i] = pickle.load(f)
    count+=1

demos = pd.DataFrame.from_dict(demos,orient='index').dropna()

# this is from Dashun's hot streak paper
df = pd.read_csv('./gs_name_profile.txt',
                 sep='\t',
                 header=None,
                 names=['index','name','work'],
                 index_col='index')

# merge and dump to csv
df = df.join(demos,how='right')
df.to_csv('./demos_pubs_merged.csv')
