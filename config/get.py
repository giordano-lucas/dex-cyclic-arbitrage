#!/usr/bin/env python
import yaml
import sys  

sys.path.append("/scratch/izar/kapps/DEX-Cyclic-Arbitrage/")

with open("/scratch/izar/kapps/DEX-Cyclic-Arbitrage/config/config.yml", "r") as ymlfile:
    cfg = yaml.safe_load(ymlfile)

