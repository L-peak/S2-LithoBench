import rasterio
href='https://copernicus-dem-30m.s3.amazonaws.com/Copernicus_DSM_COG_10_S31_00_E138_00_DEM/Copernicus_DSM_COG_10_S31_00_E138_00_DEM.tif'
with rasterio.Env(GDAL_DISABLE_READDIR_ON_OPEN='EMPTY_DIR', AWS_NO_SIGN_REQUEST='YES', GDAL_HTTP_MULTIRANGE='YES'):
    with rasterio.open(href) as src:
        print('bounds', src.bounds)
        print('crs', src.crs, 'shape', src.shape, 'res', src.res)
