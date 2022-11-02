#!/usr/bin/env python

import argparse
import logging
import multiprocessing
import time
from subprocess import Popen
import os
from typing import List, Any
import csv
import tempfile
import json
import sys

import numpy as np
import boto3
import geopandas as gpd
import rasterio
from osgeo import gdal
from shapely.geometry import Polygon
import geojson
from geojson import Polygon, Feature, FeatureCollection

BIN_DIR = os.path.dirname(__file__)


def scale(x) -> float:
    return (x - np.nanmin(x)) * (1 / (np.nanmax(x) - np.nanmin(x)) * 255)

def progress_tiff_list(filename: str) -> List[str]:
    """
    Parses tiff file names out of a csv file with inference results.
    Args:
        filename (str): Path to inference results csv file
    Returns:
        completed_tiff_files (list): List of tiff file names with inference results in input file
    """
    completed_tiff_files = []
    with open(filename, 'r') as f:
        r = csv.reader(f)
        first = True
        for row in r:
            if first:
                first = False
                continue
            completed_tiff_files.append(row[1])

    return completed_tiff_files


def polygon_inference_file_to_list(filename: str, tiff_dir: str):
    """
    Parses and typecasts csv file with inference results so that it can be read in as a list of Python objects. This
    function is for inference result files that include the string representation of the polygon object.
    Args:
        filename (str): Path to csv file with inference results
        tiff_dir (str): Path to directory containing tiff files that are listed in the input file
    Returns:
        inference_file_to_list (list): List of inference results typecast from strings to their respective Python
         objects
    """
    inference_results = []
    with open(filename, 'r') as f:
        r = csv.reader(f)
        first = True
        for row in r:
            if first:
                first = False
                continue
            polygon = row[2:-3]
            coords = []
            for j in polygon:
                coords.append(tuple([int(p) for p in j.strip('POLYGON').strip('(').strip(' (').strip(')').split(' ')]))
            inference_results.append((os.path.join(tiff_dir, row[1]), Polygon(coords), row[-3], int(row[-2]),
                                      float(row[-1])))
    return inference_results


def coords_inference_file_to_list(filename: str, tiff_dir: str):
    """
    Parses and typecasts csv file with inference results so that it can be read in as a list of Python objects. This
    function is for inference result files that includes the string representation of the polygon coordinates.
    Args:
        filename (str): Path to csv file with inference results
        tiff_dir (str): Path to directory containing tiff files that are listed in the input file
    Returns:
        inference_file_to_list (list): List of inference results typecast from strings to their respective Python
         objects
    """
    inference_file_to_list = []
    with open(filename, 'r') as f:
        r = csv.reader(f)
        first = True
        for row in r:
            if first:
                first = False
                continue
            polygon = row[2:-3]
            coords = []
            for i, val in enumerate(polygon):
                val = val.strip('(').strip(' (').strip(')').strip(' ').strip('[(').strip(')]').strip('.0')
                if i % 2 != 0:
                    coords.append((last_val, int(val)))
                else:
                    last_val = int(val)
            inference_file_to_list.append((os.path.join(tiff_dir, os.path.basename(row[1])), Polygon(coords), row[-3],
                                           int(row[-2]), float(row[-1])))
    return inference_file_to_list


def batch_list(input_list: List[Any], batches: int) -> List[List[Any]]:
    """
    Splits up an input list into a list of sub-lists. The number of sub-lists is determined by the batches argument.
    This is useful when creating batches for parallelization.
    Args:
        input_list (list): List to be broken up into sub-lists
        batches (int): Number of sub-lists to split the input_list into
    """
    f = len(input_list) // batches
    for i in range(0, len(input_list), f):
        yield input_list[i:i + f]


def write_tiff_files(input_rstr: str, name: str, tiling_dir: str, output_dir: str, geom_lookup_path: str,
                     cores: int, bucket_name: str = None, s3=None):
    """

    """
    logging.info('Starting tiff file creation')
    t1 = time.time()
    ds = gdal.Open(input_rstr)
    r = scale(ds.GetRasterBand(3).ReadAsArray()).astype('uint8')
    g = scale(ds.GetRasterBand(2).ReadAsArray()).astype('uint8')
    b = scale(ds.GetRasterBand(1).ReadAsArray()).astype('uint8')
    ds = None
    del ds

    dss = rasterio.open(input_rstr)

    output_scaled = os.path.join(output_dir,
                                 'multiband_scaled_corrected' + os.path.basename(input_rstr).split('.')[0][
                                                                -3:] + '.tiff')
    true = rasterio.open(str(output_scaled), 'w', driver='Gtiff',
                         width=dss.width, height=dss.height,
                         count=3,
                         crs=dss.crs,
                         transform=dss.transform,
                         dtype='uint8'
                         )
    true.write(r, 3)
    true.write(g, 2)
    true.write(b, 1)
    true.close()

    if bucket_name is not None and s3 is not None:
        s3.Bucket(bucket_name).upload_file(output_scaled, name + '_' + output_scaled)
    del r, g, b, true

    div = 366
    batch_space = np.linspace(0, div, cores + 1)
    batches = []
    for i in range(1, len(batch_space)):
        batches.append((int(batch_space[i-1]), int(batch_space[i])))

    output_parallel_files = os.path.join(output_dir, 'tiff_parallel_output')
    os.makedirs(output_parallel_files, exist_ok=True)

    for i, batch in enumerate(batches):
        geojson_outfile = os.path.join(output_parallel_files, f'geojson_{i}.json')
        geom_lookup_outfile = os.path.join(output_parallel_files, f'geom_lookup_{i}.json')

        Popen([sys.executable, os.path.join(BIN_DIR, 'make_tiff_files.py'), '--output_scaled', output_scaled,
               '--tiling_dir', tiling_dir, '--input_tiff', input_rstr, '--tile_start', str(batch[0]), '--tile_stop',
               str(batch[1]), '--geojson_outpath', geojson_outfile, '--geom_lookup_outpath', geom_lookup_outfile])

    # There are two files being written for each subprocess
    while len(os.listdir(output_parallel_files)) < len(batches) * 2:
        time.sleep(2 * 60)

    # Combine all of the parallel output
    geom_lookup = {}
    features = []
    for file in os.listdir(output_parallel_files):
        file_path = os.path.join(output_parallel_files, file)
        if file.startswith('geojson'):
            with open(file_path, 'r') as f:
                file_features = json.load(f)['features']
                for feature in file_features:
                    features.append(
                        {
                            "type": "Feature",
                            "properties": {},
                            "geometry": {
                                "type": "Polygon",
                                "coordinates": [feature]
                            }
                        }
                    )

        elif file.startswith('geom_lookup'):
            with open(file_path, 'r') as f:
                geoms = json.load(f)
                for geom in geoms:
                    geom_lookup[geom] = geoms[geom]

    # Write the output files
    with open(geom_lookup_path, 'w+') as f:
        json.dump({'geom_lookup': geom_lookup}, f)

    with open(os.path.join(output_dir, 'plotting_geoms.geojson'), 'w+') as f:
        crs = {
            "type": "name",
            "properties": {
                "name": str(dss.crs)
            }
        }
        geojson.dump(FeatureCollection(features, crs=crs), f)

    logging.info(f'Wrote tiff files in {time.time() - t1}s')
    print(f'Wrote tiff files to {tiling_dir}')

    return dss.crs


def perform_inference(input_rstr: str, name: str, tiling_dir: str, output_dir: str, cores: int, model_path: str,
                      crs: rasterio.crs.CRS, geom_lookup_path: str, progress_file: str = None, bucket_name: str = None,
                      s3=None):
    logging.info('Starting Inference')
    np.random.seed(42)

    # Create input and output directories for parallelization i/o
    input_parallel_files = os.path.join(output_dir, 'inference_parallel_input')
    os.makedirs(input_parallel_files, exist_ok=True)

    output_parallel_files = os.path.join(output_dir, 'inference_parallel_output')
    os.makedirs(output_parallel_files, exist_ok=True)

    t1 = time.time()

    # Create batches for parallelization
    tilling_dir_list = [os.path.join(tiling_dir, i) for i in os.listdir(tiling_dir)]
    if progress_file is not None:
        completed_tiles = progress_tiff_list(progress_file)
        uncompleted_tiling_dir_list = [i for i in tilling_dir_list if os.path.basename(i) not in completed_tiles]
        batches = list(batch_list(uncompleted_tiling_dir_list, cores))
    else:
        batches = list(batch_list(tilling_dir_list, cores))

    count = 0
    for i, batch in enumerate(batches):
        count += len(batch)
        job_path = os.path.join(input_parallel_files, f'batch_{i}_input.json')
        with open(job_path, 'w+') as f:
            json.dump({'tiling_files': batch[:10]}, f)
        out_path = os.path.join(output_parallel_files, f'batch_{i}_output.json')
        Popen([sys.executable, os.path.join(BIN_DIR, 'inference.py'), '--input_file', job_path,
               '--model_path', model_path, '--tiling_dir', tiling_dir, '--input_tiff', input_rstr,
               '--geom_lookup', geom_lookup_path, '--location_name', name, '--outpath', out_path])

    logging.info(f"Performing inference on {count} files")

    # There are three files (output, progress csv, and logfile) being written for each subprocess
    while len(os.listdir(output_parallel_files)) < len(batches) * 3:
        time.sleep(15 * 60)

    logging.info(f'Finished creating geoms list {time.time() - t1}s')
    logging.info(f'Finished creating geoms list')

    geoms = []
    for file in os.listdir(output_parallel_files):
        if not file.endswith('.json'):
            continue
        with open(os.path.join(output_parallel_files, file), 'r') as f:
            file_data = json.load(f)
            for geom in file_data['geoms']:
                geometry = Polygon(geom[1])
                geoms.append((geom[0], geometry, geom[2], geom[3], geom[4]))

    if progress_file is not None:
        geoms += polygon_inference_file_to_list(progress_file, tiff_dir=tiling_dir)

    t1 = time.time()
    file_names, geom_vals, pred_labels, pred_values, fl_values = zip(*geoms)

    geoseries_data = gpd.GeoSeries(data=geom_vals)

    test_gdff = gpd.GeoDataFrame(data=file_names, columns=['name_shp'], geometry=geoseries_data,
                                 crs=str(crs))

    test_gdff['label'] = pred_labels
    test_gdff['value'] = pred_values
    test_gdff['fl_val'] = fl_values

    logging.info(f'Finished performing inference {time.time() - t1}.s')
    logging.info('Finished performing inference')

    inference_dir = os.path.join(output_dir, 'RESNET_' + str(50) + '_Inference_' +
                                 str(input_rstr.split('.')[0][-3:]))
    test_gdff.to_file(inference_dir)

    if bucket_name is not None and s3 is not None:
        for obj in os.listdir(inference_dir):
            s3.Bucket(bucket_name).upload_file(os.path.join(inference_dir, obj), name + '_' + obj)

    logging.info('Finished uploading inference files to the S3 bucket')

    del test_gdff, pred_labels, pred_values, fl_values


def do_inference(input_rstr: str, name: str, model_path: str, progress_file: str = None, cores: int = None,
                 output_dir: str = None, s3_bucket_name: str = None):
    cores = cores if cores is not None else multiprocessing.cpu_count() - 2

    # Create output file directories
    aoi_name = 'folder_' + input_rstr.split('.')[0][-3:]
    output_dir = output_dir if output_dir is not None else tempfile.mkdtemp(prefix=f'b2p_{aoi_name}')
    tiling_dir = os.path.join(output_dir, f'Tilling_tiff_{aoi_name}_{name}')
    shape_dir = os.path.join(output_dir, f'Tilling_shp_{aoi_name}_{name}')
    os.makedirs(shape_dir, exist_ok=True)
    os.makedirs(tiling_dir, exist_ok=True)

    # Configure logging
    logfile = os.path.join(output_dir, f'output.log')
    logging.basicConfig(filename=logfile, level=logging.INFO)

    # Configure AWS s3
    s3 = boto3.resource('s3')
    bucket_name = s3_bucket_name

    geom_lookup = os.path.join(output_dir, 'tiff_geom_lookup.json')

    crs = write_tiff_files(input_rstr=input_rstr, name=name, tiling_dir=tiling_dir, output_dir=output_dir,
                           geom_lookup_path=geom_lookup, bucket_name=bucket_name, s3=s3, cores=cores)
    perform_inference(input_rstr=input_rstr, name=name, tiling_dir=tiling_dir, output_dir=output_dir, cores=cores,
                      model_path=model_path, progress_file=progress_file, crs=crs, geom_lookup_path=geom_lookup,
                      bucket_name=bucket_name, s3=s3)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()

    # Required args
    parser.add_argument('--file', '-f', type=str, required=True, help='Path to tiff file')
    parser.add_argument('--name', '-n', type=str, required=True, help='Name of the region')
    # TODO: Once structure is established give this a default value
    parser.add_argument('--model', '-m', type=str, required=True, help='Path to inference model file')

    # Optional args
    parser.add_argument('--cores', '-c', type=int, required=False, help='Number of cores to use in parallel for tiling'
                                                                        ' and inference')
    parser.add_argument('--progress_file', '-p', type=str, required=False, help='Path to progress csv file')
    parser.add_argument('--bucket_name', '-b', type=str, required=False, help='Path to AWS s3 bucket i.e. b2p.erve')
    parser.add_argument('--out_dir', '-o', type=str, required=False, help='Path to directory where output files will be'
                                                                          ' written')

    args = parser.parse_args()

    do_inference(input_rstr=args.file, name=args.name, model_path=args.model, progress_file=args.progress_file,
                 cores=args.cores, output_dir=args.out_dir, s3_bucket_name=args.bucket_name)
