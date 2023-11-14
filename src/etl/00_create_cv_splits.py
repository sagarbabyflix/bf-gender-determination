import glob
import numpy as np
import pandas as pd

from utils import create_double_cv


df = pd.read_csv('../../../train_data.csv')
imgfiles = glob.glob('../../../train_images/*')
imgfiles = [_.split('/')[-1] for _ in imgfiles]
df.columns = ['pid', 'imgfile', 'label']
label_dict = {'boy': 0, 'girl': 1, 'unable to assess': 2, 'text says boy/girl': 3}
df['label'] = df.label.map(label_dict)

df = create_double_cv(df, 'pid', 5, 5)
df.to_csv('../../../train_kfold.csv', index=False)