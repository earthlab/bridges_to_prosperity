#!/usr/bin/env python
import os
import json
import gdal
import argparse


def make_tiff_files(output_scaled: str, tiling_dir: str, input_rstr: str, tile_start: int, tile_stop: int,
                    geojson_outpath: str, geom_lookup_outpath: str):
    dem = gdal.Open(output_scaled)
    gt = dem.GetGeoTransform()
    xmin = gt[0]
    ymax = gt[3]
    res = gt[1]

    xlen = res * dem.RasterXSize
    ylen = res * dem.RasterYSize
    div = 366
    xsize = xlen / div
    ysize = ylen / div
    xsteps = [xmin + xsize * i for i in range(div + 1)]
    ysteps = [ymax - ysize * i for i in range(div + 1)]

    del gt, xmin, ymax, res, xlen, ylen
    geom_lookup = {}
    features = []
    filenames = []
    for i in range(tile_start, tile_stop):
        for j in range(tile_start, tile_stop):
            xmin = xsteps[i]
            xmax = xsteps[i + 1]
            ymax = ysteps[j]
            ymin = ysteps[j + 1]
            tiff_filename = os.path.join(tiling_dir, 'dem' + input_rstr.split('.')[0][-3:] + str(i) + '_' + str(j) +
                                         '.tif')
            filenames.append(tiff_filename)
            gdal.Warp(tiff_filename, dem, outputBounds=(xmin, ymin, xmax, ymax), dstNodata=-999)
            coords = ((xmin, ymin), (xmin, ymax), (xmax, ymax), (xmax, ymin), (xmin, ymin))
            geom_lookup[tiff_filename] = coords
            features.append([[xmin, ymin], [xmax, ymin], [xmax, ymax], [xmin, ymax], [xmin, ymin]])

            del xmin, xmax, ymax, ymin

    with open(geojson_outpath, 'w+') as f:
        json.dump({'features': features}, f)

    with open(geom_lookup_outpath, 'w+') as f:
        json.dump(geom_lookup, f)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--output_scaled', type=str, required=True, help='Path to scaled tiff file')
    parser.add_argument('--tiling_dir', type=str, required=True, help='Path to tiling directory')
    parser.add_argument('--input_tiff', type=str, required=True, help='Path to input tiff file')
    parser.add_argument('--tile_start', type=int, required=True, help='Location of the overall tile to start writing '
                                                                      'smaller tiff files at')
    parser.add_argument('--tile_stop', type=int, required=True, help='Location of the overall tile to stop writing '
                                                                     'smaller tiff files at')
    parser.add_argument('--geojson_outpath', type=str, required=True, help='Path to the geojson output json file')
    parser.add_argument('--geom_lookup_outpath', type=str, required=True,
                        help='Path to the geometry lookup output json file')
    args = parser.parse_args()
    make_tiff_files(args.output_scaled, args.tiling_dir, args.input_tiff, args.tile_start, args.tile_stop,
                    args.geojson_outpath, args.geom_lookup_outpath)
