import json

import pandas as pd
import numpy as np



def get_index():
    data_dict = {}
    info_path = '/home/szcycy/下载/VisA_20220922/split_csv/1cls.csv'
    our_path = 'visa_index.json'
    all_info = pd.read_csv(info_path)
    print(np.unique(all_info['object']),np.unique(all_info['split']),np.unique(all_info['label']))
    for obj in np.unique(all_info['object']):
        # print(all_info[(all_info['object']==obj) & (all_info['split']=='train') & ( all_info['label']=='anomaly')]['image'].values)
        data_dict[obj] = {'train':{'ok':all_info[(all_info['object']==obj) & (all_info['split']=='train') & ( all_info['label']=='normal')]['image'].values.tolist(),
                                   'ok_binary':[],
                                   'ng':[],'ng_binary':[]
                                   },
                          'test': {'ok': all_info[(all_info['object'] == obj) & (all_info['split'] == 'test') & (
                                      all_info['label'] == 'normal')]['image'].values.tolist(),
                                    'ok_binary': [],
                                    'ng': all_info[(all_info['object'] == obj) & (all_info['split'] == 'test') & (
                                                all_info['label'] == 'anomaly')]['image'].values.tolist(),
                                    'ng_binary': all_info[
                                        (all_info['object'] == obj) & (all_info['split'] == 'test') & (
                                                    all_info['label'] == 'anomaly')]['mask'].values.tolist(), }
                          }


    with open(our_path, 'w') as f:
        json.dump(data_dict, f,indent=4)
if __name__ == '__main__':
    get_index()