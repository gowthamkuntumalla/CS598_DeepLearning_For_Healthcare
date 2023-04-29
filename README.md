This is the final report for CS598 final project from: 
- Yiming Li: yl159@illinois.edu
- Gowtham Kuntumalla: gowtham4@illinois.edu

Both individuals exhibit equal impact in the project. Main contributor for each task will be labeled in parentheses.

## CONTENT:
Readmission Prediction via Deep Contextual Embedding of Clinical Concepts

## environment:
- windows 11 x64 platform, python3.7

- lasagne==0.2.dev1; theano==1.0.5; numpy==1.20.3

- And some other packages mentioned in the code
- C/C++ implementation for calculations, mianly about properly installation of g++ support. 

please refer to the final report, section 6.1 Reproduction of Original Code link:

## commands to run:


data collection stage: 

`python3.7 transfomer.py`

1. we start with a file that contains 3000 synthetic patient data, with columns: patient_id, event_date, DX_GROUP_DESCRIPTION, SERVICE_LOCATION

2. we create word bag out of DX_GROUP_DESCRIPTION

3. we label readmission into hospital out of event_date and SERVICE_LOCATION


modeling stage:

`python3.7 CONTENT.py`

Note: config.py and CONTENT.py both serve some config values /given values for parameters.

Output: files under theta_with_rnnvec folder


# Sources

## Paper Citation
Xiao C, Ma T, Dieng AB, Blei DM, Wang F (2018) Readmission prediction via deep contextual embedding of clinical concepts. PLOS ONE 13(4): e0195024. https://doi.org/10.1371/journal.pone.0195024

## Code Base Citation

Xiao, C., Ma, T., Dieng, A., Blei, D., & Wang, F. (2017). CONTENT (Version 1.0.0) [Computer software]. https://doi.org/10.1371/journal.pone.0195024