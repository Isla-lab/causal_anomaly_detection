## DESCRIPTION ##
Code for the paper:
@inproceedings{meli2024causal_anomaly,
  title={Explainable Online Unsupervised Anomaly Detection for Cyber-Physical Systems via Causal Discovery from Time Series},
  author={Meli, Daniele},
  booktitle={IEEE 20th International Conference on Automation Science and Engineering (CASE)},
  year={2024 (in publication)},
  organization={IEEE}
}

## REQUIREMENTS ##
1. Python 3.10
2. https://github.com/jakobrunge/tigramite
3. SWAT dataset available at https://itrust.sutd.edu.sg/itrust-labs_datasets/dataset_info/ (A2 version tested)
4. Pepper dataset available at https://sites.google.com/diag.uniroma1.it/robsec-data

## HOW TO RUN ##
1. Learn causal models via learn_causal.py
2. Test anomaly detection via anomaly_detection.py