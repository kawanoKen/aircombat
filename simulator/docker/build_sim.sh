#!/bin/bash -e

cd ASRCAISim1
source /opt/data/venv/acac_4th/bin/activate
pip install -U "setuptools>=62.4"
python install_all.py -i -c
python install_all.py -i -p
python install_all.py -i -u
python install_all.py -i -m ./sample/modules/R7ContestSample
