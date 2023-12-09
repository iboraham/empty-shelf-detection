# You need to run this script after running:
# -> pipenv install 
# -> pipenv shell

mim install mmengine
mim install "mmcv>=2.0.0"
git clone https://github.com/open-mmlab/mmdetection.git src/commons/mmdetection
cd src/commons/mmdetection
pip install -v -e .
cd ../../..