# MemeSTN
## Our code, data, scripts, and notebooks are available now for submission "Learning Social Meta-knowledge for Nowcasting Human Flow in Disaster".

# How to run our model?
* cd model_ours
* python traintest_MemeGCRN.py --ex=EXPERIMENT --gpu=GPU_ID
* EXPERIMENT = {typhoon-inflow, typhoon-outflow, covid-inflow, covid-outflow}

# How to run the baselines with social covariate (twitter information)?
* cd model+tw
* python traintest_MODELNAME.py --ex=EXPERIMENT --gpu=GPU_ID
* EXPERIMENT = {typhoon-inflow, typhoon-outflow, covid-inflow, covid-outflow}
* MODELNAME = {STGCN, ASTGCN, DCRNN, GraphWaveNet, DCRNN, LSTNet, AGCRN, MTGNN, GMAN, STTN}

# How to run the baselines without social covariate (only inflow/outflow)?
* cd model
* python traintest_MODELNAME.py --ex=EXPERIMENT --gpu=GPU_ID
* EXPERIMENT = {typhoon-inflow, typhoon-outflow, covid-inflow, covid-outflow}
* MODELNAME = {STGCN, ASTGCN, DCRNN, GraphWaveNet, DCRNN, LSTNet, AGCRN, MTGNN, GMAN, STTN}
