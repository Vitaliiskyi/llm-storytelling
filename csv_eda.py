# Estamating the length of the text in each row of the CSV file and 
# printing the sorted list of lengths. This can help in understanding 
# the distribution of text lengths in the dataset, which is useful for 
# preprocessing and model training.  

import pandas as pd

df = pd.read_csv('folk_tales_deduplicated.csv')
lens = []
print(df.head())
print(len(df.iloc[1].text))
for idx, row in df.iterrows():
    lens.append(len(row['text']))
    a = sorted(lens)
    print(a,end="\n\n")