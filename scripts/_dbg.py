import pandas as pd, numpy as np, rasterio
from pyproj import Transformer
from rasterio.windows import from_bounds, Window, intersection
s = pd.read_csv(r'D:\HWB\geology-paper\data\samples_mount_isa.csv')
print('utm stats', s.x_utm.min(), s.x_utm.max(), s.y_utm.min(), s.y_utm.max())
tr = Transformer.from_crs(4326,32754,always_xy=True)
xs, ys = tr.transform(s.lon.values, s.lat.values)
print('recomputed', xs.min(), xs.max(), ys.min(), ys.max())
margin=300
b=(xs.min()-margin, ys.min()-margin, xs.max()+margin, ys.max()+margin)
print('bounds', b, 'width_km', (b[2]-b[0])/1000, 'height_km', (b[3]-b[1])/1000)
href='https://sentinel-cogs.s3.us-west-2.amazonaws.com/sentinel-s2-l2a-cogs/54/K/UC/2024/8/S2B_54KUC_20240829_0_L2A/SCL.tif'
with rasterio.Env(GDAL_DISABLE_READDIR_ON_OPEN='EMPTY_DIR', AWS_NO_SIGN_REQUEST='YES', GDAL_HTTP_MULTIRANGE='YES'):
    with rasterio.open(href) as src:
        print('crs', src.crs, 'shape', src.shape, 'res', src.res, 'bounds', src.bounds)
        w = from_bounds(*b, transform=src.transform)
        print('raw window', w)
        w2 = w.round_offsets().round_lengths()
        print('rounded', w2)
        w3 = intersection(w2, Window(0,0,src.width,src.height))
        print('clipped', w3, 'shape', w3.height, w3.width)
