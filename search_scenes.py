# -*- coding: utf-8 -*-
"""
Spyder Editor

This is a temporary script file.
"""

import landsatxplore.api
import pprint

# Initialize a new API instance and get an access key
api = landsatxplore.api.API('kylecshan', '135357246Usgs')

# Request
scenes = api.search(
    dataset='LANDSAT_8_C1',
    latitude=38.7,
    longitude=-121.17,
    start_date='2018-01-01',
    end_date='2019-01-01',
    max_cloud_cover=10,
    max_results=10)

pp = pprint.PrettyPrinter(indent=4)
pp.pprint(scenes)

print('{} scenes found.'.format(len(scenes)))

for scene in scenes:
    print(scene['acquisitionDate'])

api.logout()
