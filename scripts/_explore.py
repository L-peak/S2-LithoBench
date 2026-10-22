import json
from pathlib import Path
import geopandas as gpd
import pandas as pd

DATA = Path(r'D:\HWB\geology-paper\data')

def harmonise(lith):
    if not lith: return None
    s = lith.lower()
    if 'regolith' in s: return 'regolith'
    if 'carbonate' in s: return 'carbonate'
    if 'fault' in s or 'shear' in s: return None
    if 'ultramafic' in s or 'mafic' in s: return 'mafic_igneous'
    if 'felsic' in s and 'intrusive' in s: return 'felsic_intrusive'
    if 'intermediate' in s and 'intrusive' in s: return 'felsic_intrusive'
    if 'volcanic' in s: return 'felsic_volcanic'
    if 'metasedimentary' in s or 'high grade metamorphic' in s: return 'metasedimentary'
    if any(k in s for k in ['arenite','rudite','argillaceous','detrital sediment','sedimentary siliciclastic']): return 'clastic_sedimentary'
    return None

for region in ['mount_isa','flinders']:
    gj = gpd.read_file(DATA/f'geology_{region}.geojson').set_crs(4326, allow_override=True).to_crs(32754)
    gj['cls'] = gj['lithology'].map(harmonise)
    gj['area_km2'] = gj.geometry.area/1e6
    print('==', region)
    g = gj.dropna(subset=['cls']).groupby('cls').agg(n_poly=('area_km2','size'), area_km2=('area_km2','sum'),
                                                     n_ge2=('area_km2', lambda s:(s>=2).sum()),
                                                     n_ge05=('area_km2', lambda s:(s>=0.5).sum()))
    print(g.round(1).to_string())
