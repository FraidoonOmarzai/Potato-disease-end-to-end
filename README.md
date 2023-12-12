# Potato-disease-end-to-end

## Steps:
* Create github repository and clone repository
* Define template of the project `template.py` and run it
```bash
touch template.py
python3 template.py
```
* Define setup.py file
* Create a new environment and install dependencies `requirements.txt`
```bash
conda create -n potato-env python=3.10 -y
conda activate potato-env
pip install -r requirements.txt
```
* Define logger
* Define utils/common.py