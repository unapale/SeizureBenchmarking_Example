# SeizureBenchmarking_Example

Example for validating ML algorithms using https://github.com/esl-epfl/sz-validation-framework framework

# Datasets supported 
Possible to use CHBMIT, Siena and SeizIT dataset. 

# General and personal training
Two main scripts: 
- `main_personal.py`: performs personal training using time series crossvalidation
- `main_general.py`: performs general training using leave one subject output

# Code workflow
1. Define dataset
2. Define parameters (e.g. features to be used) 
3. Input output folders defined 
4. Dataset standardization (can be performed only once)
5. Extracting annotations to standard format (can be performed only once)
6. Extracting features (and saving to new files, so that it can be performed only once). Also checks JS divergence of features during seizure and non-seizure.
7. Personal/general training. Results in annotation file with predictions. 
8. Comparison of two annotation files (true and predicted labels). Calculates and plots performance also per subject. Plots predictions in time. 


