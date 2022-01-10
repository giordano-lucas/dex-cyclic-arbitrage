#!/usr/bin/env python
import yaml
import sys,os

base_dir = '/'.join(os.getcwd().split('/')[:4])

def add_base_dir(value, acc, key):
    if not isinstance(value, dict):
        acc[key] = f"{base_dir}/{value}"
    else:
        if key is None:
            param = acc
        else:
            acc[key] = {};
            param = acc[key]
        for k,v in value.items():
            add_base_dir(v, param, k)    
    return acc

with open(f"{base_dir}/config/config.yml", "r") as ymlfile:
    raw_cfg = yaml.safe_load(ymlfile)
    cfg = add_base_dir(raw_cfg, {}, None)
    
with open(f"{base_dir}/config/models_config.yml", "r") as ymlfile:
    models_cfg = yaml.safe_load(ymlfile)


