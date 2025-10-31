"""
提取PAEsData中的经纬度
"""
import os
import pandas as pd

PAEsPath = '.\\PAEsData\\PAEsData4.csv'

data = pd.read_csv(PAEsPath, sep=',')

xy = data[['longitude','latitude']]

xy_unique = xy.drop_duplicates()

xy_unique.to_csv('.\\FeatureForTrain\\xy.csv', index=False)
xy_unique.to_csv('.\\NCData\\xy.csv', index=False)
xy_unique.to_csv('.\\PAEsData\\xy.csv', index=False)
xy_unique.to_csv('.\\SoilinfoDataNC\\xy.csv', index=False)
xy_unique.to_csv('.\\SoilinfoDataTIFF\\xy.csv', index=False)