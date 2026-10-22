import pandas as pd
p = r'D:\HWB\geology-paper\results\metrics_mount_isa.csv'
res = pd.read_csv(p)
sub = res.groupby(['feature_set','model','protocol'])[['oa','macro_f1','kappa']].mean().round(3).reset_index()
piv = sub.pivot_table(index=['feature_set','model'], columns='protocol', values='macro_f1')
piv = piv[['random','spatial']]
piv['gap'] = (piv['random']-piv['spatial']).round(3)
print(piv.sort_values('spatial', ascending=False).to_string())
print()
print('best config:', open(r'D:\HWB\geology-paper\results\best_config_mount_isa.json').read())
per = pd.read_csv(r'D:\HWB\geology-paper\results\per_class_mount_isa.csv', index_col=0)
print(per[['precision','recall','f1-score','support']].round(3).to_string())
