# QSPR_model_solubility_in_ILs
QSPR modeling using Multiclass Classification of solids temperature-dependent solubility in Ionic Liquids. This repository contains initial curated dataset and a few scripts in order to perform solubility prediction in Ionic Liquids.


**Scripts folder:**
- **rationale_test_set_selection_solubility_in_ILs.py**
  - Application of the Kennard-Stone algorithm to split the original dataset into training and test set.
- **bazingas.py**
  - asjc

**Data folder:**

After curation and physico-chemical descriptors calculation, the dataset is composed of 938 datapoints:
- **new_dataset_chemaxon_full.csv**

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
python 3.7

pandas

numpy

re

sklearn
