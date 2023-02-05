# MemeSTN
### "Learning Social Meta-knowledge for Nowcasting Human Mobility in Disaster" (WWW'23).
* Datasets Typhoon-JP, COVID-JP, and Hurricane-US are stored at ./data.
* You can check *params.txt* under each model directory for the details of experiment/data settings.  
  Note that ./model_ours/params.txt and ./model_baseline/params.txt are same.

## Requirements
* Python 3.8.8 
* pytorch 1.9.1
* pandas 1.2.4 
* numpy 1.20.1
* torch-summary 1.4.5 -> pip install torch-summary https://pypi.org/project/torch-summary/
* jpholiday -> pip install jpholiday
* holidays -> pip install holidays

## How to run our model?
* cd **model**
* python traintest_MemeSTN.py --ex=EXPERIMENT --gpu=GPU_ID  
  EXPERIMENT = {typhoon-inflow, typhoon-outflow, covid-inflow, covid-outflow, hurricane-poi}
