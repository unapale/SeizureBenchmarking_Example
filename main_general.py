from loadEeg.loadEdf import *
from parametersSetup import *
from VariousFunctionsLib import  *
from evaluate.evaluate import *
import os

#####################################################
# # SIENA DATASET
dataset='SIENA'
rootDir=  '../../../../../scratch/dan/physionet.org/files/siena-scalp-eeg/1.0.0' #when running from putty
rootDir=  '../../../../../shares/eslfiler1/scratch/dan/physionet.org/files/siena-scalp-eeg/1.0.0' #when running from remote desktop
DatasetPreprocessParams.channelNamesToKeep=DatasetPreprocessParams.channelNamesToKeep_Unipolar

# # SEIZIT DATASET
# dataset='SeizIT1'
# # rootDir=  '../../../../../databases/medical/ku-leuven/SeizeIT1/v1_0' #when running from putty
# rootDir=  '../../../../../shares/eslfiler1/databases/medical/ku-leuven/SeizeIT1/v1_0' #when running from remote desktop
# DatasetPreprocessParams.channelNamesToKeep=DatasetPreprocessParams.channelNamesToKeep_Unipolar

#CHBMIT DATASET
# dataset='CHBMIT'
# rootDir=  '../../../../../scratch/dan/physionet.org/files/chbmit/1.0.0' #when running from putty
# rootDir=  '../../../../../shares/eslfiler1/scratch/dan/physionet.org/files/chbmit/1.0.0' #when running from remote desktop
# DatasetPreprocessParams.channelNamesToKeep=DatasetPreprocessParams.channelNamesToKeep_Bipolar

#####################################################
# SET DIFFERENT PARAMETERS
# Set features to use (it will be in the ouput folder name)
FeaturesParams.featNames = np.array( ['ZeroCross'])
# FeaturesParams.featNames = np.array( ['MeanAmpl', 'LineLength'])
# FeaturesParams.featNames = np.array( ['MeanAmpl', 'LineLength','Frequency'])
# FeaturesParams.featNames = np.array( ['MeanAmpl', 'LineLength','Frequency','ZeroCross'])
# FeaturesParams.featNames = np.array( ['ZeroCrossAbs'])
FeaturesParams.featSetNames= FeaturesParams.featNames

#####################################################
# CREATE FOLDER NAMES
# Output folder for standardized dataset
outDir= '../../../10_datasets/'+ dataset+ '_Standardized'
os.makedirs(os.path.dirname(outDir), exist_ok=True)
# Output folder with calculated features
outDirFeatures= '../../../10_datasets/'+ dataset+ '_Features'
os.makedirs(os.path.dirname(outDirFeatures), exist_ok=True)
# Output folder for ML model predictions
outPredictionsFolder = '../../../10_datasets/' + dataset + '_TrainingResults/01_General_'+StandardMLParams.modelType+'_WinStep['+str(FeaturesParams.winLen)+','+str(FeaturesParams.winStep)+']_'+'-'.join(FeaturesParams.featNames)+'/'
os.makedirs(os.path.dirname(outPredictionsFolder), exist_ok=True)

# testing that folders are correct
# print(os.path.exists(rootDir))
## print(os.listdir('../../../../../'))

#####################################################
# STANDARTIZE DATASET - Only has to be done once

# # .edf as output
# standardizeDataset(rootDir, outDir) #for all datasets that are unipolar (SeizIT and Siena)
# standardizeDataset(rootDir, outDir,  origMontage='bipolar-dBanana') #for CHBMIT

#if we want to change output format
# standardizeDataset(rootDir, outDir, outFormat='csv')
# standardizeDataset(rootDir, outDir, outFormat='parquet.gzip')

# #####################################################
# EXTRACT ANNOTATIONS - Only has to be done once
if (dataset=='CHBMIT'):
    from loadAnnotations.CHBMITAnnotationConverter import *
elif (dataset == 'SIENA'):
    from loadAnnotations.sienaAnnotationConverter import *
elif (dataset=='SeizIT1'):
    from loadAnnotations.sienaAnnotationConverter import *

TrueAnnotationsFile = outDir + '/' + dataset + 'AnnotationsTrue.csv'
os.makedirs(os.path.dirname(TrueAnnotationsFile), exist_ok=True)
# annotationsTrue= convertAllAnnotations(rootDir, TrueAnnotationsFile )

# #####################################################
# # EXTRACT FEATURES AND SAVE TO FILES - Only has to be done once
# calculateFeaturesForAllFiles(outDir, outDirFeatures, DatasetPreprocessParams, FeaturesParams, outFormat ='parquet.gzip' )

# # CALCULATE KL DIVERGENCE OF FEATURES
# GeneralParams.patients = [ f.name for f in os.scandir(outDir) if f.is_dir() ]
# GeneralParams.patients.sort() #Sorting them
# FeaturesParams.allFeatNames = constructAllfeatNames(FeaturesParams)
# calculateKLDivergenceForFeatures(dataset, GeneralParams.patients , outDirFeatures, TrueAnnotationsFile, FeaturesParams)

# ####################################################
# # TRAIN GENERALIZED MODEL
#
## LOAD ALL DATA OF ALL SUBJECTS
print('LOADING ALL DATA')
# Create list of all subjects
GeneralParams.patients = [ f.name for f in os.scandir(outDir) if f.is_dir() ]
GeneralParams.patients.sort() #Sorting them
# GeneralParams.patients=GeneralParams.patients[0:3]

dataAllSubj= loadAllSubjData(dataset, outDirFeatures, GeneralParams.patients, FeaturesParams.featNames,DatasetPreprocessParams.channelNamesToKeep, TrueAnnotationsFile)

##################################
print('TRAINING') # run leave-one-subject-out CV
NonFeatureColumns= ['Subject', 'FileName', 'Time', 'Labels']
AllRes_test=np.zeros((len(GeneralParams.patients),27))
for patIndx, pat in enumerate(GeneralParams.patients):
    print(pat)
    testData= dataAllSubj[dataAllSubj['Subject'] == pat]
    trainData = dataAllSubj[dataAllSubj['Subject'] != pat]
    testDataFeatures= testData.loc[:, ~testData.columns.isin(NonFeatureColumns)]
    trainDataFeatures = trainData.loc[:, ~trainData.columns.isin(NonFeatureColumns)]

    #normalize data
    if (FeaturesParams.featNorm == 'Norm'):
        testDataFeatures= normalizeData(testDataFeatures)
        trainDataFeatures = normalizeData(trainDataFeatures)
        # (trainDataFeatures, testDataFeatures) = normalizeTrainAndTestData(trainDataFeatures, testDataFeatures)
        trainDataFeatures=removeExtremeValues(trainDataFeatures)
        testDataFeatures=removeExtremeValues(testDataFeatures)
        #remove useless feature columns
        colsToDrop=[]
        colsToDrop=removeFeaturesIfExtreme(trainDataFeatures, colsToDrop)
        colsToDrop=removeFeaturesIfExtreme(testDataFeatures, colsToDrop)
        colsToDrop=list(set(colsToDrop))
        trainDataFeatures=trainDataFeatures.drop(labels=colsToDrop, axis='columns')
        testDataFeatures=testDataFeatures.drop(labels=colsToDrop, axis='columns')

    ## STANDARD ML LEARNING
    MLstdModel = train_StandardML_moreModelsPossible(trainDataFeatures.to_numpy(), trainData['Labels'].to_numpy(), StandardMLParams)
    # MLstdModel = train_StandardML_moreModelsPossible(testDataFeatures.to_numpy(), testData['Labels'].to_numpy(), StandardMLParams)
    # testing
    (predLabels_test, probabLab_test, acc_test, accPerClass_test) = test_StandardML_moreModelsPossible(testDataFeatures.to_numpy(), testData['Labels'].to_numpy(),MLstdModel)

    # measure performance
    AllRes_test[patIndx, 0:9] = performance_sampleAndEventBased(predLabels_test, testData['Labels'].to_numpy(), PerformanceParams)
    # test smoothing - moving average
    predLabels_MovAvrg = movingAvrgSmoothing(predLabels_test, PerformanceParams.smoothingWinLen,  PerformanceParams.votingPercentage)
    AllRes_test[patIndx, 9:18] = performance_sampleAndEventBased(predLabels_MovAvrg, testData['Labels'].to_numpy(), PerformanceParams)
    # test smoothing - moving average
    predLabels_Bayes = smoothenLabels_Bayes(predLabels_test, probabLab_test, PerformanceParams.smoothingWinLen, PerformanceParams.bayesProbThresh)
    AllRes_test[patIndx, 18:27] = performance_sampleAndEventBased(predLabels_Bayes, testData['Labels'].to_numpy(), PerformanceParams)
    outputName = outPredictionsFolder + '/AllSubj_PerformanceAllSmoothing_OldMetrics.csv'
    saveDataToFile(AllRes_test, outputName, 'csv')

    #visualize predictions
    outName=outPredictionsFolder + '/'+ pat+'_PredictionsInTime'
    plotPredictionsMatchingInTime(testData['Labels'].to_numpy(), predLabels_test, predLabels_MovAvrg, predLabels_Bayes, outName, PerformanceParams)


    # Saving predicitions in time
    dataToSave = np.vstack((testData['Labels'].to_numpy(), probabLab_test, predLabels_test, predLabels_MovAvrg,  predLabels_Bayes)).transpose()   # added from which file is specific part of test set
    dataToSaveDF=pd.DataFrame(dataToSave, columns=['TrueLabels', 'ProbabLabels', 'PredLabels', 'PredLabels_MovAvrg', 'PredLabels_Bayes'])
    outputName = outPredictionsFolder + '/Subj' + pat + '_'+StandardMLParams.modelType+'_TestPredictions.csv'
    saveDataToFile(dataToSaveDF, outputName, 'parquet.gzip')

    # CREATE ANNOTATION FILE
    predlabels= np.vstack((probabLab_test, predLabels_test, predLabels_MovAvrg,  predLabels_Bayes)).transpose().astype(int)
    testPredictionsDF=pd.concat([testData[NonFeatureColumns].reset_index(drop=True), pd.DataFrame(predlabels, columns=['ProbabLabels', 'PredLabels', 'PredLabels_MovAvrg', 'PredLabels_Bayes'])] , axis=1)
    annotationsTrue=readDataFromFile(TrueAnnotationsFile)
    annotationAllPred=createAnnotationFileFromPredictions(testPredictionsDF, annotationsTrue, 'PredLabels_Bayes')
    if (patIndx==0):
        annotationAllSubjPred=annotationAllPred
    else:
        annotationAllSubjPred = pd.concat([annotationAllSubjPred, annotationAllPred], axis=0)
    #save every time, just for backup
    PredictedAnnotationsFile = outPredictionsFolder + '/' + dataset + 'AnnotationPredictions.csv'
    annotationAllSubjPred.sort_values(by=['filepath']).to_csv(PredictedAnnotationsFile, index=False)


#############################################################
#EVALUATE PERFORMANCE  - Compare two annotation files
print('EVALUATING PERFORMANCE')
labelFreq=1/FeaturesParams.winStep
TrueAnnotationsFile = outDir + '/' + dataset + 'AnnotationsTrue.csv'
PredictedAnnotationsFile = outPredictionsFolder + '/' + dataset + 'AnnotationPredictions.csv'
# Calcualte performance per file by comparing true annotations file and the one created by ML training
paramsPerformance = scoring.EventScoring.Parameters(
    toleranceStart=PerformanceParams.toleranceStart,
    toleranceEnd=PerformanceParams.toleranceEnd,
    minOverlap=PerformanceParams.minOveralp,
    maxEventDuration=PerformanceParams.maxEventDuration,
    minDurationBetweenEvents=PerformanceParams.minDurationBetweenEvents)
# performancePerFile= evaluate2AnnotationFiles(TrueAnnotationsFile, PredictedAnnotationsFile, labelFreq)
performancePerFile= evaluate2AnnotationFiles(TrueAnnotationsFile, PredictedAnnotationsFile, labelFreq, paramsPerformance)
# save performance per file
PerformancePerFileName = outPredictionsFolder + '/' + dataset + 'PerformancePerFile.csv'
performancePerFile.sort_values(by=['filepath']).to_csv(PerformancePerFileName, index=False)

# Calculate performance per subject
GeneralParams.patients = [ f.name for f in os.scandir(outDirFeatures) if f.is_dir() ]
GeneralParams.patients.sort() #Sorting them
performacePerSubj= recalculatePerfPerSubject(performancePerFile, GeneralParams.patients, labelFreq, paramsPerformance)
PerformancePerSubjName = outPredictionsFolder + '/' + dataset + 'PerformancePerSubj.csv'
performacePerSubj.sort_values(by=['subject']).to_csv(PerformancePerSubjName, index=False)
# plot performance per subject
plotPerformancePerSubj(GeneralParams.patients, performacePerSubj, outPredictionsFolder)


### PLOT IN TIME
for patIndx, pat in enumerate(GeneralParams.patients):
    print(pat)
    InName = outPredictionsFolder + '/Subj' + pat + '_' + StandardMLParams.modelType + '_TestPredictions.csv.parquet.gzip'
    data= readDataFromFile(InName)

    # visualize predictions
    outName = outPredictionsFolder + '/' + pat + '_PredictionsInTime'
    plotPredictionsMatchingInTime(data['TrueLabels'].to_numpy(), data['PredLabels'].to_numpy(), data['PredLabels_MovAvrg'].to_numpy(), data['PredLabels_Bayes'].to_numpy(), outName, PerformanceParams)


# ### FIND OPTIMAL PROCESSING PARAMETERS FOR ALL SUBJ TOGETHER
# # load all predictions in time
# TestDifferentPostprocessingParams(outPredictionsFolder, dataset, GeneralParams, StandardMLParams)