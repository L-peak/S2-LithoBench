import importlib.util, sys
spec = importlib.util.spec_from_file_location('figs', r'D:\HWB\geology-paper\scripts\06_figures.py')
m = importlib.util.module_from_spec(spec); spec.loader.exec_module(m)
m.fig_workflow()
m.fig_study_areas()
print('ok')
