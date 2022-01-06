#!/usr/bin/env python
import yaml
import sys,os

base_dir = '/'.join(os.getcwd().split('/')[:4])
with open(f"{base_dir}/config/config.yml", "r") as ymlfile:
    cfg = yaml.safe_load(ymlfile)

