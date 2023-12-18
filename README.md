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

* Get the dataset from kaggle and store it in the google drive
```
https://drive.google.com/file/d/1XboDAEEC_QsRP5HVVGPEuL27XCmPCiIH/view?usp=sharing
```


```bash
### Workflow
1. Update config.yaml
3. Update params.yaml
4. Update the entity
5. Update the configuration manager in src config
6. Update the components
7. Update the pipeline
9. Update the dvc.yaml
```

* **Data Ingestion**
* **Data Validation**
* **Model Training**