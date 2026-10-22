import pandas as pd, numpy as np
for r in ['mount_isa','flinders']:
    df = pd.read_csv(fr'D:\HWB\geology-paper\data\features_{r}.csv')
    print('==', r, df.shape)
    print(df['lith_class'].value_counts().to_string())
    cols = ['B02','B04','B08','B11','B12','ndvi','iron_oxide','clay','dem_elev','dem_slope','tex_B08']
    print(df[cols].describe().T[['mean','std','min','max']].round(3).to_string())
    print('NaNs:', int(df[cols].isna().sum().sum()), ' valid_dates mean:', df.n_valid_dates.mean().round(2))
