import os
import subprocess as sp
import getpass
import pathlib
from typing import List, Union
from datetime import datetime, timedelta
from argparse import Namespace
import multiprocessing as mp
import tqdm

import geopandas as gpd
from shapely.geometry import Polygon
import boto3
import numpy as np
import mgrs


PACKAGE_PATH = os.path.join(os.path.dirname(__file__))


def _download_task(namespace: Namespace):
    s3 = boto3.client('s3')
    s3.download_file(namespace.bucket_name, namespace.available_file,
                     os.path.join(namespace.outdir, namespace.available_file.replace('/', '_')),
                     ExtraArgs={'RequestPayer': 'requester'}
                     )


class SentinelAPI:
    SAML2AWS_INSTALL_ERROR_STR = "Error configuring saml2aws. Is it installed? Check for installation by running" \
                                 " 'saml2aws version'\n If not installed follow instructions at " \
                                 "https://curc.readthedocs.io/en/iaasce-954_grouper/cloud/aws/getting-started/" \
                                 "aws-cli-saml2aws.html"
    MAX_SESSION_DURATION = 7200
    DEFAULT_SENTINEL_MASTER_PATH = os.path.join(PACKAGE_PATH, 'data', 'sentinel_master',
                                                'kml_sentinelmy_new_sentinel.shp')

    def __init__(self, identikey: str):
        self._identikey = identikey
        self._password = getpass.getpass('identikey password:')
        self._configure_saml2aws()
        self._generate_session_token()
        self._s3 = boto3.client('s3')
        self._bucket_name = 'sentinel-s2-l1c'

    def download(self, bounds: List[float], buffer: float, outdir: str, start_date: str, end_date: str):
        # Convert the buffer from meters to degrees lat/long at the equator
        buffer /= 111000

        # Adjust the bounding box to include the buffer (subtract from min lat/long values, add to max lat/long values)
        bounds[0] -= buffer
        bounds[1] -= buffer
        bounds[2] += buffer
        bounds[3] += buffer

        os.makedirs(outdir, exist_ok=True)

        available_files = self._find_available_files(bounds, start_date, end_date)
        total_data = 0
        for file in available_files:
            total_data += file[1]
        total_data /= 1E9

        args = []
        for file in available_files:
            if '/preview/' in file[0]:
                continue
            args.append(Namespace(available_file=file[0], bucket_name=self._bucket_name, outdir=outdir))

        proceed = input(f'Found {len(args)} files for download. Total size of files is'
                        f' {round(total_data, 2)}GB and estimated cost will be ${round(0.09 * total_data, 2)}'
                        f'. Proceed (y/n)?')

        if proceed == 'y':
            with mp.Pool(mp.cpu_count() - 1) as pool:
                for _ in tqdm.tqdm(pool.imap_unordered(_download_task, args), total=len(args)):
                    pass

    def _find_available_files(self, bounds, start_date: str, end_date: str):
        ref_date = self._str_to_datetime(start_date)
        date_paths = []
        while ref_date <= self._str_to_datetime(end_date):
            tt = ref_date.timetuple()
            date_paths.append(f'/{tt.tm_year}/{tt.tm_mon}/{tt.tm_mday}/')
            ref_date = ref_date + timedelta(days=1)

        info = []
        mgrs_grids = self._find_overlapping_mgrs(bounds)

        # TODO: Start out by just doing this the dumb way and make a request for something that might not even be there.
        #  It may be more wise to find which links are available first
        for grid_string in mgrs_grids:
            grid = f'tiles/{grid_string[:2]}/{grid_string[2]}/{grid_string[3:5]}'
            response = self._s3.list_objects_v2(
                Bucket=self._bucket_name,
                Prefix=grid + '/',
                MaxKeys=300,
                RequestPayer='requester'
            )
            if 'Contents' not in list(response.keys()):
                continue

            for date in date_paths:
                response = self._s3.list_objects_v2(
                    Bucket=self._bucket_name,
                    Prefix=grid + date + '0/',
                    MaxKeys=100,
                    RequestPayer='requester'
                )
                if 'Contents' in list(response.keys()):
                    info += [(v['Key'], v['Size']) for v in response['Contents'] if
                             'B02.jp2' in v['Key'] or 'B03.jp2' in v['Key'] or 'B04.jp2' in v['Key']]

        return info

    @staticmethod
    def _find_overlapping_mgrs(bounds) -> List[str]:
        mgrs_object = mgrs.MGRS()
        mgrs_strings = []

        def _classify_points(lat: Union[float, np.array], lon: Union[float, np.array]):
            variable_lat = not isinstance(lat, float)
            variable_coords = lat if variable_lat else lon
            for coord in variable_coords:
                grid_name = mgrs_object.toMGRS(coord if variable_lat else lat, coord if not variable_lat else lon,
                                               MGRSPrecision=1)[:5]
                if grid_name not in mgrs_strings:
                    mgrs_strings.append(grid_name)

        lat_inc = abs((bounds[3] - bounds[1]) / 10)
        lon_inc = abs((bounds[2] - bounds[0]) / 10)

        # Traverse from bottom left to top left in intervals of 0.1 degree of latitude
        _classify_points(lon=bounds[0], lat=np.arange(bounds[1], bounds[3] + lat_inc, lat_inc)[:-1])

        # Traverse from top left to top right in intervals of 0.1 degree of longitude
        _classify_points(lat=bounds[3], lon=np.arange(bounds[0], bounds[2] + lon_inc, lon_inc)[:-1])

        # Traverse from bottom right to top right in intervals of 0.1 degree of latitude
        _classify_points(lon=bounds[2], lat=np.arange(bounds[1], bounds[3] + lat_inc, lat_inc)[:-1])

        # Traverse from bottom left to bottom right in intervals of 0.1 degree of longitude
        _classify_points(lat=bounds[1], lon=np.arange(bounds[0], bounds[2] + lon_inc, lon_inc)[:-1])

        return mgrs_strings

    def _str_to_datetime(self, date: str):
        return datetime.strptime(date, '%Y-%m-%d')

    # TODO: A lot of this could be moved to a base API class for use with any AWS S3 bucket. Also it is unnecessary if
    #  running this code from an ec2 instance with the proper IAM profile
    def _generate_session_token(self, ttl: int = 7200):
        self._configure_session_ttl(ttl)
        try:
            sp.call([
                'saml2aws', 'login', f'--username={self._identikey}', f'--password={self._password}', '--skip-prompt'
            ])

            with open(os.path.join(pathlib.Path().home(), '.aws', 'credentials'), 'r') as f:
                start = False

                f1 = False
                f2 = False
                f3 = False
                for line in f.readlines():
                    if line == '[saml]\n':
                        start = True
                    if not start:
                        continue

                    if line.startswith('aws_access_key_id'):
                        os.environ['AWS_ACCESS_KEY_ID'] = str(line.split('= ')[1].strip('\n'))
                        f1 = True
                    elif line.startswith('aws_secret_access_key'):
                        os.environ['AWS_SECRET_ACCESS_KEY'] = str(line.split('= ')[1].strip('\n'))
                        f2 = True
                    elif line.startswith('aws_security_token'):
                        os.environ['AWS_SESSION_TOKEN'] = str(line.split('= ')[1].strip('\n'))
                        f3 = True

                    if f1 and f2 and f3:
                        break

            self._s3 = boto3.client(
                's3',
                aws_access_key_id=os.environ['AWS_ACCESS_KEY_ID'],
                aws_secret_access_key=os.environ['AWS_SECRET_ACCESS_KEY'],
                aws_session_token=os.environ['AWS_SESSION_TOKEN']
            )

        except FileNotFoundError:
            print(self.SAML2AWS_INSTALL_ERROR_STR)
            return

    def _configure_session_ttl(self, ttl: int):
        if ttl > self.MAX_SESSION_DURATION:
            print(f'Requested TTL exceeds max of {self.MAX_SESSION_DURATION}. Using max duration.')
            ttl = self.MAX_SESSION_DURATION

        saml2aws_path = os.path.join(pathlib.Path.home(), '.saml2aws')
        if not os.path.exists(saml2aws_path):
            print(f'Could not find .saml2aws file at {saml2aws_path}. Is saml2aws installed? If not installed follow'
                  f' instructions at "\
                  "https://curc.readthedocs.io/en/iaasce-954_grouper/cloud/aws/getting-started/aws-cli-saml2aws.html"')
            return

        with open(saml2aws_path, 'r') as f:
            lines = f.readlines()

        for i, line in enumerate(lines):
            if line.startswith('aws_session_duration'):
                lines[i] = f'aws_session_duration    = {ttl}\n'

        with open(saml2aws_path, 'w') as f:
            f.writelines(lines)

        print(f'Set ttl to {ttl} seconds')

    def _configure_saml2aws(self):
        try:
            sp.call([
                'saml2aws', 'configure', '--idp-provider=ShibbolethECP', '--mfa=push',
                '--url=https://fedauth.colorado.edu/idp/profile/SAML2/SOAP/ECP',
                f'--username={self._identikey}', f'--password={self._password}', '--skip-prompt'
            ])
        except FileNotFoundError:
            print(self.SAML2AWS_INSTALL_ERROR_STR)
