# -*- coding: utf-8 -*-
"""
Created on Mon Sep 30 17:50:44 2019

@author: kcsii_000
"""

from landsatxplore.earthexplorer import EarthExplorer

ee = EarthExplorer('kylecshan', '135357246Usgs')

ee.download(scene_id='LC80430332018177LGN00', output_dir='./data')

ee.logout()