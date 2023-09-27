# SeizureBenchmarking_Example

Example for validating ML algorithms using [framework](https://github.com/esl-epfl/sz-validation-framework). All description of the framework is on https://eslweb.epfl.ch/epilepsybenchmarks/.

### Datasets supported 
Possible to use CHBMIT, Siena and SeizIT dataset. 

### General and personal training
Two main scripts: 
- `main_personal.py`: performs personal training using time series crossvalidation
- `main_general.py`: performs general training using leave one subject output
- `main_KFoldGeneral.py`: performs general training using K fold crossvalidation (needed for big datasets, e.g. SeizIT)

### ML models implemented (but not all tested)
KNN, SVM, DT, RF, AdaBoost

### Features that can be extracted 
Calculation of several types of features is implemented. Each feature class is saved into separate file once they are created so that they can later be easy combined and loaded depending on needs fot the training. Feature classes supported: 
- `MeanAmpl`: mean amplitude, 1 feature 
- `LineLength`: line length,1 feature 
- `Frequency`: Absolute and relative frequency spectrum, 17 features ('p_dc_rel', 'p_mov_rel', 'p_delta_rel', 'p_theta_rel', 'p_alfa_rel', 'p_middle_rel', 'p_beta_rel', 'p_gamma_rel', 'p_dc', 'p_mov', 'p_delta', 'p_theta', 'p_alfa', 'p_middle', 'p_beta', 'p_gamma', 'p_tot')
- `ZeroCross`: Zero crossing features with signal approximation (from the [paper](https://iopscience.iop.org/article/10.1088/1741-2552/aca1e4)). Number of features depends on number of ZeroCross threshold parameters (ZC_thresh_arr). Usually it is 5 or 6 features. 


### Code workflow
1. Define dataset
2. Define parameters (e.g. features to be used) 
3. Input output folders defined 
4. Dataset standardization (can be performed only once)
5. Extracting annotations to standard format (can be performed only once)
6. Extracting features (and saving to new files, so that it can be performed only once). Also checks JS divergence of features during seizure and non-seizure.
7. Personal/general training. Results in annotation file with predictions. 
8. Comparison of two annotation files (true and predicted labels). Calculates and plots performance also per subject. Plots predictions in time. 


