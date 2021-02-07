# QSPR_model_solubility_in_ILs
QSPR modeling using Multiclass Classification of solids temperature-dependent solubility in Ionic Liquids. This repository contains initial curated dataset and a few scripts in order to perform solubility prediction in Ionic Liquids.


**Scripts folder:**
- **rationale_test_set_selection_solubility_in_ILs.py**
  - Application of the Kennard-Stone algorithm to split the original dataset into training and test set.
- **MultiClass_Random_Forest_Solubility.py**
  - Descriptor importance calculation, Proximity matrix computation, Solubility prediction and statistical assessment

**Data folder:**

After curation and physico-chemical descriptors calculation:
- **new_dataset_chemaxon_full.csv**

Training set with descriptors and solubility class (0 - Soluble, 1 - Moderately Soluble, 2 - Insoluble):
- **training_set_final.csv**

Test set with descriptors and solubility class (0 - Soluble, 1 - Moderately Soluble, 2 - Insoluble):
- **test_set_final.csv**

External test set descriptors:
- **x_ext_ts_scaled.csv**

External test set solubility class:
- **y_ext_set_final.csv**

## Corresponding Authors:

João Miguel Inês (j.ines@fct.unl.pt)

## Credits:

Associated Lab in Sustainable Chemistry (LAQV-REQUIMTE)

NOVA University of Lisbon - Cheminformatics Lab

## Reference:

Please, cite the paper when you use the data or the scripts:

Inês, João; Klimenko, Kyrylo; Aires-de-Sousa, João; Esperança, José; Carrera, Gonçalo (2021) QSPR modeling of the solubility curve of solids in ionic liquids. JOURNAL. 

DOI

## Dependencies:
- python: 3.7
- pandas: 0.24.2
- numpy: 1.16.4
- re: 2.2.1
- sklearn: 0.20.3
- scipy: 1.3.0
- sklearn: 0.20.3 
- IPython: 7.10.1
