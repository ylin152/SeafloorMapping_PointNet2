import os
import pandas as pd
import numpy as np

# read a list of data to be processed
rlist_path = './3.csv'

rlist = []
dir = './data_8192/111'

with open(rlist_path, 'r', encoding='utf-8-sig') as f_obj:
    for line in f_obj:
        rlist.append(line.rstrip('\n'))

print(rlist[:10])

for line in rlist:
    list_spilt = line.split(',')
    list_spilt = list(filter(None, list_spilt))
    file = list_spilt[0] + '.txt'
    file_path = os.path.join(dir, file)
    df = pd.read_csv(file_path, sep=' ', header=None, names=['lon', 'lat', 'x', 'y', 'elev', 'signal_conf', 'label'])
    if len(list_spilt) == 1:
        df.loc[df['label'] == 1, 'label'] = 0
    # else:
    #     for c in list_spilt[1:]:
    #         if '<' in c:
    #             c = c.replace('<', '')
    #             c = int(c)
    #             df.loc[df['lat'] < c, 'label'] = 0
    #         elif '>' in c:
    #             c = c.replace('>', '')
    #             c = int(c)
    #             df.loc[df['lat'] > c, 'label'] = 0
    #         else:
    #             c1 = c.split('-')[0]
    #             c2 = c.split('-')[1]
    #             c1 = int(c1)
    #             c2 = int(c2)
    #             df.loc[(df['lat'] > c1) & (df['lat'] < c2), 'label'] = 0

    df.to_csv(file_path, sep=' ', header=False, index=False)

#
# for line in list:
#     for file, min, max in line:
#         file_path = os.path.join(dir, file)
#         df = pd.read_table(file_path, sep=' ', header=None)
#
#         assert min != max
#
#         if min == -1:
#             df[df[:, 1] < max, -1] = 0
#         elif max == -1:
#             df[df[:, 1] > min, -1] = 0
#         else:
#             df[(df[:, 1] > min & df[:, 1] < max), -1] = 0


