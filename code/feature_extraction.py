import numpy as np
import pandas as pd

'''
Input data must be a excel sheet, containing kinematics of only one impact 
The data in sheet should be in the shape of n*n, no header
Column should be in the following order: 
LinAccX LinAccY LinAccZ LinAccRes RotVelX RotVelY RotVelZ RotVelRes RotAccX RotAccY RotAccZ RotAccRes t(ms)
The duration of impact (number of rows) does not matter
'''

num_feature = 2


def feature(files, dof, ):
    ftrs = np.zeros(shape=(len(files), len(dof), num_feature))  # shape = impact * channel * feature
    for impact_id, f in enumerate(files):
        df = pd.read_excel(f)
        ftrs[impact_id] = feature_extraction(df, dof)
    return ftrs


def feature_extraction(df, dof):
    feature_matrix = np.zeros(shape=(len(dof), num_feature))  # shape = channel * feature
    for idx, col in enumerate(df.columns[[x for x in dof]]):
        signal = df[col].to_numpy()
        max = np.max(signal)
        min = np.min(signal)
        delta = max - min

        # power of max
        max_power_sqrt = np.sqrt(abs(max))  # square root of abs

        # write feature into list
        feature_list = [delta, max_power_sqrt]

        for i in range(len(feature_list)):
            feature_matrix[idx][i] = feature_list[i]

    return feature_matrix

