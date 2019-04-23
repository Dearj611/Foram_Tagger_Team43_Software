import os
import pandas as pd
# This file builds the csv files for our dataset, which allows us to later perform cross-fold validation

def build_files(path, num_of_files=4, store_path='./data-csv'):
    '''
    path: image directory (can be of varying depth)
    '''
    num_of_species = len(os.listdir(path))
    builder = {i:[] for i in range(num_of_files)}
    for dirpath, dirs, filename in os.walk(path):
        if dirs != []:
            continue
        total_files = len(filename)
        split = int(total_files/num_of_files)
        complete_path = [os.path.join(os.path.basename(dirpath), name) for name in filename]
        for i in range(num_of_files-1):
            x = builder[i]
            x += complete_path[i*split:(i+1)*split]
            builder[i] = x
        x = builder[num_of_files-1]
        x += complete_path[(num_of_files-1)*split:]
        builder[num_of_files-1] = x
    for i in range(1, num_of_files):
        assert(builder[0] != builder[i])
    for key, value in builder.items():
        df = pd.DataFrame([[name, os.path.dirname(name)] for name in value])
        df = df.sample(frac=1).reset_index(drop=True)   # shuffles the rows
        df.columns = ['location', 'species']
        filepath = os.path.join(store_path, 'file'+str(key)+'.csv')
        df.to_csv(filepath, encoding='utf-8', index=False)
        print(len(df.species.unique()))
        print(len(df))
        assert(len(df.species.unique()) == num_of_species)


def create_master_csv(path):
    builder = []
    for dirpath, dirs, filename in os.walk(path):
        if dirs != []:
            continue
        complete_path = [[os.path.join(os.path.basename(dirpath), name), os.path.basename(dirpath)] for name in filename]
        builder += complete_path
        df = pd.DataFrame(builder)
        df.columns = ['location', 'species']
        df.to_csv('./data-csv/master.csv', encoding='utf-8', index=False)


def validate(path, number_of_files):
    '''
    ensures that the number of splits is possible
    '''
    total = [len(filename) for _,_,filename in os.walk(path)]
    total.pop(0)
    if number_of_files > min(total):
        print('incomplete split')


# build_files('./training-images', store_path='./data-csv')
create_master_csv('./training-images')