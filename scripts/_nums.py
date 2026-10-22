import pandas as pd
for r in ['mount_isa','flinders']:
    print('=====', r)
    m = pd.read_csv(fr'D:\HWB\geology-paper\results\metrics_summary_{r}.csv', header=[0,1,2], index_col=[0,1,2])
    sel = m.loc[(slice(None), slice(None), slice(None)), :]
    df = m.reset_index()
    df.columns = ['feature_set','model','protocol','oa_mean','oa_std','f1_mean','f1_std','kap_mean','kap_std']
    sub = df[(df.model.isin(['RF','ET','HGB'])) & (df.feature_set.isin(['S','SI','SIT','SITT']))]
    print(sub.pivot_table(index=['feature_set'],columns=['model','protocol'],values='f1_mean').round(3).to_string())
    per = pd.read_csv(fr'D:\HWB\geology-paper\results\per_class_{r}.csv', index_col=0)
    print('per-class F1:', per['f1-score'].round(3).to_dict())
    imp = pd.read_csv(fr'D:\HWB\geology-paper\results\importance_{r}.csv').head(6)
    print('top features:', [(a, round(b,4)) for a,b in zip(imp.feature, imp.importance)])
