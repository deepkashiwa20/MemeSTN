# MemeSTN
## Our code, data, scripts, and notebooks are available now for submission "Learning Social Meta-knowledge for Nowcasting Human Flow in Disaster".

* The directory is structured in a flat style and only with two levels. The traffic datasets are stored in DATA directories (e.g., METRLA), and the python files are put in workDATA directories (e.g., workMETRLA). Entering the work directory for a certain dataset, we can find MODEL class file (e.g., DCRNN.py) and its corresponding running program named pred\_MODEL.py (e.g., pred\_DCRNN.py). We can run ``python MODEL.py'' to simply check the model architecture without feeding the training data and run ``python pred\_MODEL.py'' to train and test the model. Additionally, Param.py file contains a variety of hyper-parameters as described in Section 5.1 that allow the experiment to be customized in a unified way. Metrics.py file contains the metric functions listed in Section 5.1. Utils.py file integrates a set of supporting functions such as pickle file reader and self-defined loss function. More details about the usability and implementation can be found at GitHub.

## How to run our model?
* cd model+tw
* python traintest_MegaCRN.py
* DATASET = {METRLA, PEMSBAY, Shuto-Expy, Electricity}

## How to run the baselines without social covariate (only inflow/outflow)?
* cd model
* python traintest_MODELNAME.py --ex=EXPERIMENT --gpu=GPU_ID
* EXPERIMENT = {typhoon-inflow, typhoon-outflow, covid-inflow, covid-outflow}
* MODELNAME = {STGCN, ASTGCN, DCRNN, GraphWaveNet, DCRNN, LSTNet, AGCRN, MTGNN, GMAN, STTN}
