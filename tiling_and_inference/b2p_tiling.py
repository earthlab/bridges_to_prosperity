#!/usr/bin/env python

import argparse
import logging
import multiprocessing
import time
from subprocess import Popen

import boto3
import geopandas as gpd
import rasterio
from fastai.vision import *
from osgeo import gdal
from shapely.geometry import Polygon


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


class Tiling:
    def __init__(self, input_rstr: str, name: str, progress_file: str = None, cores: int = None,
                 output_dir: str = None, s3_bucket_name: str = None, progress_file: str = None):
        self._input_rstr = input_rstr
        self._name = name
        self._progress_file = progress_file
        self._cores = cores if cores is not None else multiprocessing.cpu_count() - 2
        self._progress_file = progress_file

        # Create output file directories
        aoi_name = 'folder_' + self._input_rstr.split('.')[0][-3:]
        self._output_dir = output_dir if output_dir is not None else tempfile.mkdtemp(prefix=f'b2p_{aoi_name}')
        self._tilling_dir = os.path.join(self._output_dir, f'Tilling_tiff_{aoi_name}_{self._name}')
        self._shape_dir = os.path.join(self._output_dir, f'Tilling_shp_{aoi_name}_{self._name}')
        os.makedirs(self._shape_dir, exist_ok=True)
        os.makedirs(self._tilling_dir, exist_ok=True)

        # Configure logging
        logfile = os.path.join(self._output_dir, f'output.log')
        logging.basicConfig(filename=logfile, level=logging.INFO)

        #Configure AWS s3
        self._s3 = boto3.resource('s3')
        self._bucket_name = s3_bucket_name

    @staticmethod
    def _batch_list(input_list, batches):
        f = len(input_list) // batches
        for i in range(0, len(input_list), f):
            yield input_list[i:i + f]

    @staticmethod
    def _generate_jobfile_name(parallel_dir: str):
        ls = os.listdir(parallel_dir)

        if not ls:
            return f'{random.randint(1, 10000)}.json'

        job_file = ls[0]
        while job_file in ls:
            job_file = f'{random.randint(1, 10000)}.json'

        return job_file


    def write_tiff_files(self):
        logging.info('Starting tiff file creation')
        t1 = time.time()
        ds = gdal.Open(self._input_rstr)
        r = scale(ds.GetRasterBand(3).ReadAsArray()).astype('uint8')
        g = scale(ds.GetRasterBand(2).ReadAsArray()).astype('uint8')
        b = scale(ds.GetRasterBand(1).ReadAsArray()).astype('uint8')
        ds = None
        del ds

        dss = rasterio.open(self._input_rstr)

        output_scaled = os.path.join(self._output_dir,
            'multiband_scaled_corrected' + os.path.basename(self._input_rstr).split('.')[0][-3:] + '.tiff')
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

        if self._bucket_name is not None:
            self._s3.Bucket(self._bucket_name).upload_file(output_scaled, self._name + '_' + output_scaled)

        del r, g, b, dss, true

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

        for i in range(div):
            for j in range(div):
                xmin = xsteps[i]
                xmax = xsteps[i + 1]
                ymax = ysteps[j]
                ymin = ysteps[j + 1]
                gdal.Warp(os.path.join(self._tilling_dir, 'dem' + self._input_rstr.split('.')[0][-3:] + str(i) + str(j)
                                       + '.tif'), dem, outputBounds=(xmin, ymin, xmax, ymax), dstNodata=-999)
                del xmin, xmax, ymax, ymin
        del dem, div, xsteps, ysteps
        logging.info(f'Wrote tiff files in {time.time() - t1}s')
        print(f'Wrote tiff files to {self._tilling_dir}')

    def write_shape_files(self) -> str:

        # Get the coordinate reference system of the tile
        rds = rioxarray.open_rasterio(os.path.join(self._tilling_dir, os.path.join(self._tilling_dir)[3]))
        gdf_crs = rds.rio.crs
        del rds

        # Create a temp working directory for parallelization i/o
        input_parallel_files = os.path.join(self._output_dir, 'shape_file_parallel_input')
        os.makedirs(input_parallel_files, exist_ok=True)

        # Make a unique directory in parallel directory
        output_parallel_files = os.path.join(self._output_dir, 'shape_file_parallel_output')
        os.makedirs(output_parallel_files, exist_ok=True)

        logging.info('Starting shape file creation')

        # Batch the input files for parallelization
        tilling_dir_list = [os.path.join(self._tilling_dir, file) for file in os.listdir(self._tilling_dir)]
        logging.info(f'Length of tilling dir: {len(os.listdir(self._tilling_dir))}')
        batches = list(self._batch_list(tilling_dir_list, self._cores))

        # Create the shape files in parallel
        t1 = time.time()
        for i, batch in enumerate(batches):
            job_path = os.path.join(input_parallel_files, f'batch_{i}_input.json')
            with open(job_path, 'w+') as f:
                json.dump({'tiling_files': batch}, f)
            outpath = os.path.join(output_parallel_files, f'batch_{i}_output.json')
            Popen([sys.executable, os.path.join(BIN_DIR, 'make_shape_files.py'), '--input_file', job_path,
                   '--shape_dir', self._shape_dir, '--crs', gdf_crs.to_string(), '--outpath', outpath])

        # Wait until all the job files have been written out
        while len(os.listdir(self._shape_dir)) < len(tilling_dir_list) * 5:
            print('SLEEPING')
            time.sleep(3 * 60)

        file_bounds = []
        for file in os.listdir(output_parallel_files):
            with open(os.path.join(output_parallel_files, file), 'r') as f:
                f_data = json.load(f)
                for bound in f_data['file_bounds']:
                    file_bounds.append((bound[0], Polygon(bound[1])))

        logging.info(f'Made tiff and shape files in {time.time() - t1}s using {len(batches)} cores')
        logging.info('Finished making tile tiff and shape files')

        names, bounds = zip(*file_bounds)
        geoseries = gpd.GeoSeries(data=bounds)
        rdf = gpd.GeoDataFrame(data=names, columns=['name_shp'], geometry=geoseries, crs=gdf_crs)

        shape_tiles = os.path.join(self._output_dir, 'shp_tiles_' + str(self._input_rstr.split('.')[0][-3:]))
        rdf.to_file(shape_tiles)
        logging.info('Finished making tile shp files')

        if self._bucket_name is not None:
            for obj in os.listdir(shape_tiles):
                self._s3.Bucket(self._bucket_name).upload_file(os.path.join(shape_tiles, obj), self._name + '_' + obj)

        del rdf
        logging.info('Finished uploading tile shp files to the S3 bucket')

        return shape_tiles

    def perform_inference(self, shape_tiles, model_path):
        logging.info('Starting Inference')
        test_gdf = gpd.read_file(
            os.path.join(self._output_dir, 'shp_tiles_' + str(self._input_rstr.split('.')[0][-3:]),
                         'shp_tiles_' + str(self._input_rstr.split('.')[0][-3:]) + '.shp')
        )
        np.random.seed(42)

        # Create input and output directories for parallelization i/o
        input_parallel_files = os.path.join(self._output_dir, 'inference_parallel_input')
        os.makedirs(input_parallel_files, exist_ok=True)

        output_parallel_files = os.path.join(self._output_dir, 'inference_parallel_output')
        os.makedirs(output_parallel_files, exist_ok=True)

        t1 = time.time()

        # Create batches for parallelization
        tilling_dir_list = [os.path.join(self._tilling_dir, i) for i in os.listdir(self._tilling_dir)]
        if self._progress_file is not None:
            completed_tiles = progress_tiff_list(self._progress_file)
            uncompleted_tiling_dir_list = [i for i in tilling_dir_list if os.path.basename(i) not in completed_tiles]
            batches = list(self._batch_list(uncompleted_tiling_dir_list, self._cores))
        else:
            batches = list(self._batch_list(tilling_dir_list, self._cores))

        count = 0
        for i, batch in enumerate(batches):
            count += len(batch)
            job_path = os.path.join(input_parallel_files, f'batch_{i}_input.json')
            with open(job_path, 'w+') as f:
                json.dump({'tiling_files': batch}, f)
            outpath = os.path.join(output_parallel_files, f'batch_{i}_output.json')
            args = [sys.executable, os.path.join(BIN_DIR, 'inference.py'), '--input_file', job_path,
                   '--model_path', model_path,
                   '--shape_path', os.path.join(shape_tiles, 'shp_tiles_' + str(self._input_rstr.split('.')[0][-3:]) + '.shp'),
                   '--tiling_dir', self._tilling_dir, '--input_tiff', self._input_rstr, '--location_name', self._name,
                   '--outpath', outpath]

            Popen(args)

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

        if self._progress_file is not None:
            geoms += self._progress_to_list(self._progress_file, tiff_dir=self._tilling_dir)

        t1 = time.time()
        file_names, geom_vals, pred_labels, pred_values, fl_values = zip(*geoms)

        geoseries_data = gpd.GeoSeries(data=geom_vals)

        test_gdff = gpd.GeoDataFrame(data=file_names, columns=['name_shp'], geometry=geoseries_data,
                                     crs=test_gdf.crs)

        test_gdff['label'] = pred_labels
        test_gdff['value'] = pred_values
        test_gdff['fl_val'] = fl_values

        logging.info(f'Finished performing inference {time.time() - t1}.s')
        logging.info('Finished performing inference')

        inference_dir = os.path.join(self._output_dir, 'RESNET_' + str(50) + '_Inference_' +
                                     str(self._input_rstr.split('.')[0][-3:]))
        test_gdff.to_file(inference_dir)

        if self._bucket_name is not None:
            for obj in os.listdir(inference_dir):
                self._s3.Bucket(self._bucket_name).upload_file(os.path.join(inference_dir, obj), self._name + '_' + obj)

        logging.info('Finished uploading inference files to the S3 bucket')

        del test_gdf, test_gdff, pred_labels, pred_values, fl_values


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--name', '-n', type=str, required=True, help='Name of the region')
    parser.add_argument('--file', '-f', type=str, required=True, help='Path to tiff file')
    parser.add_argument('--cores', '-c', type=int, required=False, help='Number of cores to use in parallel for tiling'
                                                                        ' and inference')
    parser.add_argument('--progress_file', '-p', type=str, required=False, help='Path to progress csv file')
    parser.add_argument('--bucket_name', '-b', type=str, required=True, help='Path to AWS s3 bucket i.e. b2p.erve')
    args = parser.parse_args()

    tiling(args.file, args.name, HOME_DIR, args.progress_file, args.cores)