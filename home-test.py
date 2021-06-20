import math
import os
import subprocess
import pandas as pd
import numpy as np

import matplotlib.pyplot as plt
import seaborn as sns

# Runtime environment/requirement for running code
# 
# *-memory
#       description: System memory
#       physical id: 0
#       size: 8GiB
# *-cpu
#       product: Intel(R) Core(TM) i3-2350M CPU @ 2.30GHz
#       vendor: Intel Corp.
#       physical id: 1
#       bus info: cpu@0
#       size: 916MHz
#       capacity: 2300MHz
# 
# =============================
# 
# No LSB modules are available.
# Distributor ID:	Ubuntu
# Description:	Linux Lite 5.4
# Release:	20.04
# Codename:	focal
# 
# =============================
# 
# python 3.8.5 64 bit
# numpy==1.20.3
# pandas==1.2.4
# seaborn==0.11.1
# matplotlib==3.4.2
# ipython==7.24.1
# ipython-genutils==0.2.0

DATA_PATH = 'hash_catid_count.csv'
cat_dict_frequency = {}
cat_dict_count = {}

# I. Data processing

def load_embedding_file():
    
    frequency = -1
    key_frequency = -1
    count = -1
    key_count = -1

    # Read data 
    with open(DATA_PATH, 'r') as embed_file:
        for line in embed_file:
            content = line.strip().split()

            categories = content[1]
            counts = content[2]

            categories_arr = categories[1:len(categories) - 1].split(',')
            counts_arr = counts[1:len(counts) - 1].split(',')
            for i in range(len(categories_arr)):
                if categories_arr[i] not in cat_dict_frequency:
                    cat_dict_frequency[categories_arr[i]] = 1
                    cat_dict_count[categories_arr[i]] = int(counts_arr[i])
                else:
                    cat_dict_frequency[categories_arr[i]] = cat_dict_frequency[categories_arr[i]] + 1
                    cat_dict_count[categories_arr[i]] = cat_dict_count[categories_arr[i]] + int(counts_arr[i])

    for key in cat_dict_frequency:
        if frequency < cat_dict_frequency[key]:
            frequency = cat_dict_frequency[key]
            key_frequency = key

        if count < cat_dict_count[key]:
            count = cat_dict_count[key]
            key_count = key

    return key_frequency, key_count, cat_dict_frequency[key_frequency], cat_dict_count[key_count]

def create_bar_chart():
    categories_f_pd = pd.DataFrame(cat_dict_frequency.items(), columns=['ObjectId', 'Frequency'])
    categories_f_pd.set_index('ObjectId', inplace=True)
    categories_f_pd.index = categories_f_pd.index.astype(float)

    categories_c_pd = pd.DataFrame(cat_dict_count.items(), columns=['ObjectId', 'Count'])
    categories_c_pd.set_index('ObjectId', inplace=True)
    categories_c_pd.index = categories_c_pd.index.astype(float)

    # filter (optional)
    categories_f_pd = categories_f_pd[categories_f_pd['Frequency'] > 1000]
    categories_c_pd = categories_c_pd[categories_c_pd['Count'] > 1000]

    fig, (axs1, axs2) = plt.subplots(1, 2)

    axs1.set_ylabel('Distribution of Object-id, by Frequency') 
    sns.barplot(x=categories_f_pd.index, y=categories_f_pd['Frequency'], ax=axs1)

    axs2.set_ylabel('Distribution of Object-id, by Count') 
    sns.barplot(x=categories_c_pd.index, y=categories_c_pd['Count'], ax=axs2)

    plt.show()

key_f, key_c, frequency, count = load_embedding_file()


# What is the most popular category for this sample ? (highest frequency - term is defined in section II)
print('object-id: {} with highest frequency: {}'.format(key_f, frequency))

# Which category has the largest appeared times ? (the category having the total largest counter in the sample data)
print('object-id: {} with total largest counter: {}'.format(key_c, count))

# Is there any idea how to represent/visualize the sample data for analysing ?
create_bar_chart()



# II. Data analysing

# Calculate the frequency for each category in the sample.
for key in cat_dict_frequency:
    print('category-id: {} with frequency: {}'.format(key, cat_dict_frequency[key]))

# From the calculation above, what do you think about the data sample ?
# 
# Overall, a lot of categories have frequencies less than 1000 (about 50%, 
# I guess that it doesn't have much more value), so difficult to show on the chart. 
# I think we can filter to get the category with more value than another.


# III. Algorithm
# Sort the file by object_id (preferred example code than pseudo-code and processing diagram)


## split the file into 100M pieces named fileChunkNNNN
subprocess.call('split -b100M {} fileChunk'.format(DATA_PATH), shell=True)
## Sort each of the pieces and delete the unsorted one
subprocess.call('for f in fileChunk*; do sort "$f" > "$f".sorted && rm "$f"; done', shell=True)
## merge the sorted files    
subprocess.call('sort -T /sort/ --parallel=4 -muo file_sort.csv -k 1,3 fileChunk*.sorted', shell=True)