''' Library including various functions for ML for epilepsy detection'''
__author__ = "Una Pale"
__email__ = "una.pale at epfl.ch"
import sys
import os
sys.path.insert(1, os.path.join(sys.path[0],'..'))
import glob
import math
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
from entropy import *
import scipy
import pyedflib
from sklearn import svm
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import BaggingClassifier
from sklearn.ensemble import AdaBoostClassifier
import scipy.io
from scipy import signal
import pandas as pd
import scipy.io
import datetime
from timescoring.annotations import Annotation
from timescoring import scoring
from timescoring import visualization

def bandpower(x, fs, fmin, fmax):
    ''' Function that calculates energy of specific frequency band of FFT spectrum
    Args:
        x: Signal
        fs: Sampling frequency
        fmin: Min frequency
        fmax: Max frequency
    Returns: Bandpower spectrum
    '''
    f, Pxx = scipy.signal.periodogram(x, fs=fs)
    ind_min = scipy.argmax(f > fmin) - 1
    ind_max = scipy.argmax(f >fmax) - 1
    return scipy.trapz(Pxx[ind_min: ind_max+1], f[ind_min: ind_max+1])

def calculateFeaturesOneDataWindow_Frequency(data,  samplFreq):
    ''' Function that calculates various frequency features relevant for epileptic seizure detection.
    From paper: D. Sopic, A. Aminifar, and D. Atienza, e-Glass: A Wearable System for Real-Time Detection of Epileptic Seizures, 2018

    Args:
        data: Data from which to calcualte bandpower.
        samplFreq: Sampling frequency of the signal.

    Returns: 16 values of absolute and relative bandpowers.
    '''

    #band power
    p_tot = bandpower(data, samplFreq, 0,  45)
    p_dc = bandpower(data, samplFreq, 0, 0.5)
    p_mov = bandpower(data, samplFreq, 0.1, 0.5)
    p_delta = bandpower(data, samplFreq, 0.5, 4)
    p_theta = bandpower(data, samplFreq, 4, 8)
    p_alfa = bandpower(data, samplFreq, 8, 13)
    p_middle = bandpower(data, samplFreq, 12, 13)
    p_beta = bandpower(data, samplFreq, 13, 30)
    p_gamma = bandpower(data, samplFreq, 30, 45)
    p_dc_rel = p_dc / p_tot
    p_mov_rel = p_mov / p_tot
    p_delta_rel = p_delta / p_tot
    p_theta_rel = p_theta / p_tot
    p_alfa_rel = p_alfa / p_tot
    p_middle_rel = p_middle / p_tot
    p_beta_rel = p_beta / p_tot
    p_gamma_rel = p_gamma / p_tot

    featuresAll= [p_dc_rel, p_mov_rel, p_delta_rel, p_theta_rel, p_alfa_rel, p_middle_rel, p_beta_rel, p_gamma_rel,
             p_dc, p_mov, p_delta, p_theta, p_alfa, p_middle, p_beta, p_gamma, p_tot]
    return (featuresAll)

def calculateMLfeatures_oneCh(X, DatasetPreprocessParams, FeaturesParams, type):
    ''' Function that calculate feature of interest for specific signal.
    It discretizes signal into windows and calculates feature(s) for each window.

    Args:
        X: Original raw data.
        DatasetPreprocessParams: Class with various preprocessing paremeters.
        FeaturesParams: Class with various feature parameters.
        type: Name of feature type to calculate.

    Returns: Matrix with calcualted features and their names.
    '''

    segLenIndx = int(FeaturesParams.winLen * DatasetPreprocessParams.samplFreq)  # length of EEG segments in samples
    slidWindStepIndx = int( FeaturesParams.winStep * DatasetPreprocessParams.samplFreq)  # step of slidin window to extract segments in samples
    index = np.arange(0, len(X) - segLenIndx, slidWindStepIndx).astype(int)
    for i in range(len(index)):
        sig = X[index[i]:index[i] + segLenIndx]

        if (type == 'MeanAmpl'):
            featVal = np.mean(np.abs(np.copy(sig)))
            numFeat = 1
            allFeatNames =FeaturesParams.indivFeatNames_MeanAmpl
        elif (type == 'LineLength'):
            featVal = np.mean(np.abs(np.diff(np.copy(sig))))
            numFeat = 1
            allFeatNames =FeaturesParams.indivFeatNames_LL
        elif (type == 'Frequency'):
            featVal=calculateFeaturesOneDataWindow_Frequency(np.copy(sig), DatasetPreprocessParams.samplFreq)
            numFeat = len(featVal)
            allFeatNames =FeaturesParams.indivFeatNames_Freq

        if (i==0):
            featureValues=np.zeros((len(index), numFeat))
        featureValues[i,:]=featVal

    # allFeatNames= constructAllfeatNames(FeaturesParams)
    return (featureValues,allFeatNames)

def polygonal_approx(arr, epsilon):
    '''     Performs an optimized version of the Ramer-Douglas-Peucker algorithm assuming as an input
    an array of single values, considered consecutive points, and **taking into account only the
    vertical distances**.

    Args:
        arr: Raw signal
        epsilon: Threshold parameter for the approximation.

    Returns: Approximated signal.

    '''
    def max_vdist(arr, first, last):
        """
        Obtains the distance and the index of the point in *arr* with maximum vertical distance to
        the line delimited by the first and last indices. Returns a tuple (dist, index).
        """
        if first == last:
            return (0.0, first)
        frg = arr[first:last+1]
        leng = last-first+1
        dist = np.abs(frg - np.interp(np.arange(leng),[0, leng-1], [frg[0], frg[-1]]))
        idx = np.argmax(dist)
        return (dist[idx], first+idx)

    if epsilon <= 0.0:
        raise ValueError('Epsilon must be > 0.0')
    if len(arr) < 3:
        return arr
    result = set()
    stack = [(0, len(arr) - 1)]
    while stack:
        first, last = stack.pop()
        max_dist, idx = max_vdist(arr, first, last)
        if max_dist > epsilon:
            stack.extend([(first, idx),(idx, last)])
        else:
            result.update((first, last))
    return np.array(sorted(result))

def zero_crossings(arr):
    """Returns the positions of zero-crossings in the derivative of an array, as a binary vector"""
    return np.diff(np.sign(np.diff(arr))) != 0

def calculateMovingAvrgMeanWithUndersampling(data, winLen, winStep):
    ''' Calculates moving average over data
    Args:
        data: Signal to process
        winLen: Length of window for average calulation. In number of samples.
        winStep: Length of steps in which to move window. In number of samples.
    Returns: Moving average of given data
    '''
    kernel = np.ones(winLen) / winLen
    data_convolved = np.convolve(data, kernel, mode='same')  # calculating averga
    data_convolved2 =data_convolved[2*winStep:-2*winStep:winStep]
    return (data_convolved2)

def calculateZCfeatures_oneCh(sig, DatasetPreprocessParams, FeaturesParams):
    ''' Feature that calculates zero-cross features for signal of one channel

    Args:
        sig: Data signal
        DatasetPreprocessParams: Class with various preprocessing parameters
        FeaturesParams: Class with various feature parameters

    Returns:
        zeroCrossFeaturesAll: Calculate zero cross values
        actualThrValues: Actual threshold values used
        featNames: Feature names

    '''
    if (FeaturesParams.ZC_thresh_type=='abs'):
        ZCthrs= FeaturesParams.ZC_thresh_arr
    else:
        ZCthrs= FeaturesParams.ZC_thresh_arr_rel
    numFeat=len(FeaturesParams.ZC_thresh_arr)+1
    actualThrValues=np.zeros((numFeat-1))

    #calculate signal range
    sigRange = np.percentile(sig, 95) - np.percentile(sig, 5)

    '''Zero-crossing of the original signal, counted in 1-second continuous sliding window'''
    x = np.convolve(zero_crossings(sig), np.ones(DatasetPreprocessParams.samplFreq), mode='same')
    featVals= calculateMovingAvrgMeanWithUndersampling(x, int(DatasetPreprocessParams.samplFreq * FeaturesParams.winLen), int( DatasetPreprocessParams.samplFreq * FeaturesParams.winStep))
    zeroCrossFeaturesAll = np.zeros((len(featVals), numFeat ))
    zeroCrossFeaturesAll[:, 0]=featVals

    featNames=list(['ZCThr0'])
    for EPSthrIndx, EPSthr in enumerate(ZCthrs):
        if (FeaturesParams.ZC_thresh_type == 'abs'):
            actualThrValues[EPSthrIndx] = EPSthr #FOR ABSOLUTE THRESHOLDS
        else:
            actualThrValues[ EPSthrIndx]=EPSthr*sigRange # FOR RELATIVE THRESHOLDS
        if (actualThrValues[ EPSthrIndx]==0): #cannot be 0
            actualThrValues[EPSthrIndx]=0.1
            print('actualThrValues[ EPSthrIndx]=0!')
        featNames.append('ZCThr'+str(EPSthr))
        # Signal simplification at the given threshold, and zero crossing count in the same way
        sigApprox = polygonal_approx(sig, epsilon=actualThrValues[EPSthrIndx] )#!!!! NEW TO HAVE RELATIVE THRESHOLDS
        sigApproxInterp = np.interp(np.arange(len(sig)), sigApprox, sig[sigApprox])
        x = np.convolve(zero_crossings(sigApproxInterp), np.ones(DatasetPreprocessParams.samplFreq),  mode='same')
        zeroCrossFeaturesAll[:,  EPSthrIndx + 1] = calculateMovingAvrgMeanWithUndersampling(x, int(DatasetPreprocessParams.samplFreq * FeaturesParams.winLen), int(DatasetPreprocessParams.samplFreq * FeaturesParams.winStep))

    return(zeroCrossFeaturesAll, actualThrValues, featNames)


def calculateFeaturesFromFile(data, DatasetPreprocessParams, FeaturesParams, fName):
    ''' Calculates all features in one file. Previously also filters signal.

    Args:
        data: Data to calculate features from.
        DatasetPreprocessParams: Class with various preprocessing parameters.
        FeaturesParams: Class with various feature parameters.
        fName: Feature type to calculate.

    Returns: Dataframe with all calculated features.

    '''
    X=data.to_numpy()
    sos = signal.butter(4, [1, 20], 'bandpass', fs=DatasetPreprocessParams.samplFreq, output='sos')

    for ch, chName in enumerate(data.columns):
        sig = X[:, ch]
        sigFilt = signal.sosfiltfilt(sos, sig)  # filtering
        # sigFilt=sig
        if ('ZeroCross' in fName):
            (featVals, _, featNames) = calculateZCfeatures_oneCh(np.copy(sigFilt),DatasetPreprocessParams, FeaturesParams)
        else:
            (featVals, featNames) = calculateMLfeatures_oneCh(sigFilt, DatasetPreprocessParams,  FeaturesParams, fName)

        if (ch == 0):
            AllFeatures = featVals
            featChNames = [chName + '-' + s for s in featNames]
        else:
            AllFeatures = np.hstack((AllFeatures, featVals))
            featChNames = featChNames + [chName + '-' + s for s in featNames]

    featuresDF=pd.DataFrame(AllFeatures, columns= featChNames)
    return featuresDF

def readEdfFile(fileName):
    ''' Reads .edf file and returns  data[numSamples, numCh], sampling frequency, names of channels'''

    f = pyedflib.EdfReader(fileName)
    n = f.signals_in_file
    channelNames = f.getSignalLabels()
    samplFreq =f.getSampleFrequency(0)
    #read start time
    fileStartTime=datetime.datetime(f.startdate_year, f.startdate_month, f.startdate_day, f.starttime_hour, f.starttime_minute, f.starttime_second, f.starttime_subsecond)
    #read data ch by ch
    data = np.zeros((f.getNSamples()[0], n))
    for i in np.arange(n):
        data[:, i] = f.readSignal(i)
    dataDF=pd.DataFrame(data, columns=channelNames)

    return (dataDF, samplFreq ,fileStartTime)

def createTimeStamps(fileStartTime, samplFreq, lenfile, winLen, stepLen):
    ''' Creates time stamps for every sample of the file.

    Args:
        fileStartTime: File start time.
        samplFreq: Original ampling frequency of data.
        lenfile: Length of file.
        winLen:  Length of one window from which features were calculated.
        stepLen: How often data was calculated from each winLen. Moving steps of winLen long windows.

    Returns: Time stamps of data.

    '''
    index = np.arange(0, lenfile- int(winLen*samplFreq),  int(stepLen*samplFreq))/samplFreq #.astype(int)
    timeStamps=[]
    for i in index:
        timeStamps.append(fileStartTime + datetime.timedelta(seconds=i))
    return timeStamps


def calculateFeaturesForAllFiles(folderIn, folderOut, DatasetPreprocessParams, FeaturesParams, outFormat ='parquet.gzip' ):
    ''' Loads one by one file, filters data and calculates different features, each of them in individual files so that
    later they can be easier chosen and combined
    Args:
    folderIn (str): root directory of the raw data in standard format
    folderOut (str): output directory of the calculated features
    DatasetPreprocessParams (class): class with preprocessing parameters
    FeaturesParams (class): class with feature parameters
    outFormat (str, optional): format in which to save output data. Data is dataframe with columns (Time, Feature name...)
    '''


    edfFiles = np.sort(glob.glob(os.path.join(folderIn, '**/*.edf'), recursive=True))
    for edfFile in edfFiles:
        print(edfFile)
        eegDataDF, samplFreq , fileStartTime= readEdfFile(edfFile)  # Load data
        # Create time stamps for each sample
        timeStamps= createTimeStamps(fileStartTime, samplFreq, eegDataDF.shape[0], FeaturesParams.winLen, FeaturesParams.winStep)

        outFile = folderOut + edfFile[len(folderIn):]  # Swap rootDir for outDir
        os.makedirs(os.path.dirname(outFile), exist_ok=True) # Create directory for file

        for fIndx, fName in enumerate(FeaturesParams.featNames):
            if (fName=='ZeroCrossAbs'):
                FeaturesParams.ZC_thresh_type='abs'
            else: #defauls is rel
                FeaturesParams.ZC_thresh_type = 'rel'
            # Calculate features
            featuresDF= calculateFeaturesFromFile(eegDataDF, DatasetPreprocessParams, FeaturesParams, fName)
            # Add time information
            toSave = pd.concat([pd.DataFrame(timeStamps, columns=['Time']), featuresDF], axis=1)

            if(outFormat == 'parquet.gzip'):  # parquet gzip
                toSave.to_parquet(outFile[:-4] + '-'+ fName+'.'+ outFormat, index=False, compression='gzip')
            elif (outFormat == 'csv.gzip'):
                toSave.to_csv(outFile[:-4] + '-'+ fName+'.'+ outFormat, index=False, compression='gzip')
            elif (outFormat == 'csv'):
                toSave.to_csv(outFile[:-4] + '-'+ fName+'.'+ outFormat, index=False)

def readDataFromFile( inputName):
    ''' Reads data from given file. Supports several type of files: 'csv', 'csv.gzip', 'parquet.gzip'

    Args:
        inputName: Name of the file to read
    Returns: Dataframe with data from the file
    '''
    if ('.csv.gzip' in inputName):
        df= pd.read_csv(inputName, compression='gzip')
    elif ('.parquet.gzip' in inputName):
        df= pd.read_parquet(inputName)
    else:
        df= pd.read_csv(inputName)
    # data=df.to_numpy()
    return (df)

def saveDataToFile( data,  outputName, type):
    ''' Saves data to a file. Supports several type of files: 'csv', 'csv.gzip', 'parquet.gzip'

    Args:
        data: Numpy matrix or Dataframe of data to save
        outputName: Output name of a file
        type: Extension of file to save. Supports several type of files: 'csv', 'csv.gzip', 'parquet.gzip'
    '''
    # if ('.csv' not in outputName):
    #     outputName= outputName+'.csv'
    if ~isinstance(data, pd.DataFrame):
        df = pd.DataFrame(data=data)
    if (type=='gzip'): #csv gzip
        df.to_csv(outputName + '.gz', index=False, compression='gzip')
    elif (type == 'parquet' or type=='parquet.gzip'): #parquet gzip
        df.to_parquet(outputName + '.parquet.gzip', index=False, compression='gzip')
    else: #csv
        df.to_csv(outputName, index=False)

# def concatenateDataFromFiles(fileNames):
#     ''' loads and concatenates data from all files in file name list
#     creates array noting lengths of each file to know when new file started in appeded data'''
#     startIndxOfFiles=[]
#     for f, fileName in enumerate(fileNames):
#         data_df = readDataFromFile(fileName)
#         if (f==0):
#             dataOut=data_df
#             startIndxOfFiles.append(data_df.shape[0])
#         else:
#             dataOut = pd.concat([dataOut, data_df])
#             startIndxOfFiles.append(startIndxOfFiles[-1]+ data_df.shape[0])
#     return (dataOut, startIndxOfFiles)


# def concatenateDataFromFiles_oldFeatures(fileNames, labelsFile):
#     ''' loads and concatenates data from all files in file name list
#     creates array noting lengths of each file to know when new file started in appeded data'''
#     annotations_df = readDataFromFile(labelsFile)
#     filesList=annotations_df.filepath.to_list()
#
#     startIndxOfFiles=[]
#     for f, fileName in enumerate(fileNames):
#         # Create names to match files
#         dir=os.path.dirname(fileName).split('/')[-1]
#         fn=os.path.basename(fileName).split('.')[0]
#         fn2 = fn.split('_')[0]+'_'+fn.split('_')[1]
#         filePathLabels=fn2+'_Labels.parquet.gzip'
#         # filePathToSearch=dir+'/'+fn+'.edf'
#         # indx=np.array(filesList.index(filePathToSearch)).reshape((1,-1))
#
#         # Read data
#         data_df = readDataFromFile(fileName)
#         try:
#             labels_df= readDataFromFile(os.path.dirname(fileName)+'/'+filePathLabels)
#         except:
#             filePathLabels = fn2 + '_s_Labels.parquet.gzip'
#             labels_df = readDataFromFile(os.path.dirname(fileName) + '/' + filePathLabels)
#         labels=labels_df.to_numpy().squeeze()
#
#
#         #Concatenate all files
#         if (f==0):
#             dataOut=data_df
#             # startIndxOfFiles.append(data_df.shape[0])
#             labelsOut=labels
#             subjOut=['chb'+fn2[4:6]] * data_df.shape[0]
#             fileOut = [fn2] * data_df.shape[0]
#         else:
#             dataOut = pd.concat([dataOut, data_df])
#             # startIndxOfFiles.append(startIndxOfFiles[-1]+ data_df.shape[0])
#             labelsOut=np.concatenate((labelsOut, labels), axis=0)
#             subjOut = np.concatenate((subjOut, ['chb'+fn2[4:6]] * data_df.shape[0]), axis=0)
#             fileOut = np.concatenate((fileOut, [fn2] * data_df.shape[0]), axis=0)
#
#     #add Labels column to dataframe
#     dataOut.insert(0,'Labels',labelsOut.astype(int))
#     # add subject
#     dataOut.insert(0, 'Subject', subjOut)
#     dataOut.insert(1, 'FileName', fileOut)
#
#     #remove every 2nt row beceause here step was 0.5Hz
#     dataOut=dataOut.iloc[::2, :]
#     return (dataOut)

def concatenateDataFromFilesWithLabels(dataset, fileNames, labelsFile):
    '''   Loads and concatenates data from all files in fileNames list and also adds labels from labelsFile with annotations.

    Args:
        dataset: Dataset which is used, to account for differences in file names
        fileNames: List of files to load
        labelsFile: Name of annotation file with labels

    Returns: Dataframe containing all feature data of given subject. Creates corresponding labels from annotations file.
    Colums of dataframe are: 'Subject', 'FileName', 'Time', 'Labels', all features...
    '''
    annotations_df = readDataFromFile(labelsFile)
    filesList=annotations_df.filepath.to_list()

    startIndxOfFiles=[]
    for f, fileName in enumerate(fileNames):
        # Create names to match files
        dir=os.path.dirname(fileName).split('/')[-1]
        #remove extension
        # fn=os.path.basename(fileName).split('.')[0]
        if ('.parquet.gzip' in fileName):
            fn=os.path.basename(fileName)[0:-13]
        elif ('.gzip' in fileName):
            fn=os.path.basename(fileName)[0:-5]
        elif ('.csv' in fileName):
            fn = os.path.basename(fileName)[0:-4]
        if (dataset=='CHBMIT'):
            fn1 = fn.split('-')[0]
            filePathToSearch=dir+'/'+fn1+'.edf'
        elif (dataset=='SIENA' or dataset=='siena' or dataset=='Siena'): #for siena dataset
            fn1 = fn.split('-')[0]+'-'+fn.split('-')[1]
            filePathToSearch=dir+'/'+fn1+'.edf'
        try:
            indx=np.array(filesList.index(filePathToSearch)).reshape((1,-1))
        except:
            print('a')

        # Read data
        data_df = readDataFromFile(fileName)

        # Create labels for this file
        labels=np.zeros(data_df.shape[0])
        for i in indx:
            i=i.squeeze()
            if ('sz' in annotations_df.event[i]):
                t1 = datetime.timedelta(seconds=int(annotations_df.startTime[i]))+ datetime.datetime.strptime(annotations_df.dateTime[i],  "%Y-%m-%d %H:%M:%S")
                t2 = datetime.timedelta(seconds=int(annotations_df.endTime[i])) + datetime.datetime.strptime(annotations_df.dateTime[i],  "%Y-%m-%d %H:%M:%S")
                indxRangeNum=(data_df.Time>=t1).to_numpy()*1 + (data_df.Time>=t2).to_numpy()*1
                indxsRange=np.where(indxRangeNum==1)
                labels[indxsRange]=1

        #Concatenate all files
        if (f==0):
            dataOut=data_df
            # startIndxOfFiles.append(data_df.shape[0])
            labelsOut=labels
            subjOut=[dir] * data_df.shape[0]
            fileOut = [filePathToSearch] * data_df.shape[0]
        else:
            dataOut = pd.concat([dataOut, data_df])
            # startIndxOfFiles.append(startIndxOfFiles[-1]+ data_df.shape[0])
            labelsOut=np.concatenate((labelsOut, labels), axis=0)
            subjOut = np.concatenate((subjOut, [dir] * data_df.shape[0]), axis=0)
            fileOut = np.concatenate((fileOut, [filePathToSearch] * data_df.shape[0]), axis=0)

    #add Labels column to dataframe
    dataOut.insert(1,'Labels',labelsOut.astype(int))
    # add subject
    dataOut.insert(0, 'Subject', subjOut)
    dataOut.insert(1, 'FileName', fileOut)

    return (dataOut)

def removeExtremeValues(data):
    ''' Removes +int, - inf and nan values and replaces them with means of columns (features)
    Args:
        data: Dataframe containg data (each features is one column)
    Returns: Data with extreme values replaced with the mean of the column
    '''
    # replace nan values with mean of columns
    data = data.replace([np.inf, -np.inf], np.nan)
    numeric_cols = data.select_dtypes(include=['float64', 'int64']).columns
    for colName in numeric_cols:
        data[colName] = data[colName].replace(np.nan, data[colName].mean())
    return data

def removeFeaturesIfExtreme(data, colsToDrop):
    ''' Removes whole feature column from data if it is all the time 0 or nan
    Args:
        data: Dataframe containing data (each features is one column)
        colsToDrop: Column (feature) names to drop, returns updated versioun of it
    Returns: Updated version of columns to drop from dataFrame. They have to be dropped outside of this function.
    '''
    numeric_cols = data.select_dtypes(include=['float64', 'int64']).columns
    for colName in numeric_cols:
        if (data[colName].mean()==np.nan or data[colName].sum()==0): #if all 0 or all nan remove column
            # data.drop(labels=colName)
            colsToDrop.append(colName)
    return colsToDrop

def normalizeData(data):
    ''' Normalizes data to range from 0 to 1. Performs it column (feature) wise.
    Args:
        data: Dataframe containing all features
    Returns: Normalized dataframe
    '''
    data0ut=data.copy()
    numeric_cols = data.select_dtypes(include=['float64', 'int64']).columns
    for colName in numeric_cols:
        data0ut[colName] = (data[colName]- data[colName].min()) / ( data[colName].max() - data[colName].min() )
    return data0ut

def normalizeTrainAndTestData(dataTrain, dataTest):
    ''' Normalizes at the same time train and test data, but with min/max values from train dataset to be more fair.

    Args:
        dataTrain: Dataframe cointaining all train feature values
        dataTest: Dataframe cointaining all test feature values
    Returns: Normalized train and test data
    '''
    dataTrain0ut=dataTrain.copy()
    dataTest0ut = dataTest.copy()
    numeric_cols = dataTrain.select_dtypes(include=['float64', 'int64']).columns
    for colName in numeric_cols:
        dataTest0ut[colName] = (dataTest0ut[colName] - dataTrain0ut[colName].min()) / ( dataTrain0ut[colName].max() - dataTrain0ut[colName].min())
        dataTrain0ut[colName] = (dataTrain0ut[colName]- dataTrain0ut[colName].min()) / ( dataTrain0ut[colName].max() - dataTrain0ut[colName].min() )
    return (dataTrain0ut, dataTest0ut)

def train_StandardML_moreModelsPossible(X_train, y_train,  StandardMLParams):
    ''' Functions that  trains model using (X_train, y_train) data.
    It supports several types of standard ML models whose parameters are passed in StandardMLParams class.
    ML models currently implemented are KNN, SVM, Decision tree,  Random forest, Bagging and AdaBoost.

    Args:
        X_train: Data (numpy matrix) to train the model on
        y_train: Labels (numpy array) to train the model on
        StandardMLParams: Class with many parameters for ML models
    Returns: Trained ML model
    '''

    #MLmodels.modelType = 'KNN'
    if (StandardMLParams.modelType=='KNN'):
        model = KNeighborsClassifier(n_neighbors=StandardMLParams.KNN_n_neighbors, metric=StandardMLParams.KNN_metric)
        model.fit(X_train, y_train)
    elif (StandardMLParams.modelType=='SVM'):
        model = svm.SVC(kernel=StandardMLParams.SVM_kernel, C=StandardMLParams.SVM_C, gamma=StandardMLParams.SVM_gamma)
        model.fit(X_train, y_train)
    elif (StandardMLParams.modelType=='DecisionTree' or 'DT'):
        if (StandardMLParams.DecisionTree_max_depth==0):
            model = DecisionTreeClassifier(random_state=0, criterion=StandardMLParams.DecisionTree_criterion, splitter=StandardMLParams.DecisionTree_splitter)
        else:
            model = DecisionTreeClassifier(random_state=0, criterion=StandardMLParams.DecisionTree_criterion, splitter=StandardMLParams.DecisionTree_splitter,  max_depth=StandardMLParams.DecisionTree_max_depth)
        model = model.fit(X_train, y_train)
    elif (StandardMLParams.modelType=='RandomForest' or 'RF'):
        if (StandardMLParams.DecisionTree_max_depth == 0):
            model = RandomForestClassifier(random_state=0, n_estimators=StandardMLParams.RandomForest_n_estimators, criterion=StandardMLParams.DecisionTree_criterion ) #, min_samples_leaf=10
        else:
            model = RandomForestClassifier(random_state=0, n_estimators=StandardMLParams.RandomForest_n_estimators, criterion=StandardMLParams.DecisionTree_criterion,  max_depth=StandardMLParams.DecisionTree_max_depth) #, min_samples_leaf=10
        model = model.fit(X_train, y_train)
    elif (StandardMLParams.modelType=='BaggingClassifier'):
        if (StandardMLParams.Bagging_base_estimator=='SVM'):
            model = BaggingClassifier(base_estimator=svm.SVC(kernel=StandardMLParams.SVM_kernel, C=StandardMLParams.SVM_C, gamma=StandardMLParams.SVM_gamma), n_estimators=StandardMLParams.Bagging_n_estimators,random_state=0)
        elif  (StandardMLParams.Bagging_base_estimator=='KNN'):
            model = BaggingClassifier(base_estimator= KNeighborsClassifier(n_neighbors=StandardMLParams.KNN_n_neighbors, metric=StandardMLParams.KNN_metric), n_estimators=StandardMLParams.Bagging_n_estimators,random_state=0)
        elif (StandardMLParams.Bagging_base_estimator == 'DecisionTree'):
            model = BaggingClassifier(DecisionTreeClassifier(random_state=0, criterion=StandardMLParams.DecisionTree_criterion, splitter=StandardMLParams.DecisionTree_splitter),
                n_estimators=StandardMLParams.Bagging_n_estimators, random_state=0)
        model = model.fit(X_train, y_train)
    elif (StandardMLParams.modelType=='AdaBoost'):
        if (StandardMLParams.Bagging_base_estimator=='SVM'):
            model = AdaBoostClassifier(base_estimator=svm.SVC(kernel=StandardMLParams.SVM_kernel, C=StandardMLParams.SVM_C, gamma=StandardMLParams.SVM_gamma), n_estimators=StandardMLParams.Bagging_n_estimators,random_state=0)
        elif  (StandardMLParams.Bagging_base_estimator=='KNN'):
            model = AdaBoostClassifier(base_estimator= KNeighborsClassifier(n_neighbors=StandardMLParams.KNN_n_neighbors, metric=StandardMLParams.KNN_metric), n_estimators=StandardMLParams.Bagging_n_estimators,random_state=0)
        elif (StandardMLParams.Bagging_base_estimator == 'DecisionTree'):
            model = AdaBoostClassifier(DecisionTreeClassifier(random_state=0, criterion=StandardMLParams.DecisionTree_criterion, splitter=StandardMLParams.DecisionTree_splitter),
                n_estimators=StandardMLParams.Bagging_n_estimators, random_state=0)
        model = model.fit(X_train, y_train)

    return (model)



def test_StandardML_moreModelsPossible(data,trueLabels,  model):
    ''' Gives predictions for using trained model. Returns predictions and probability.
    Aso calculates simple overall accuracy and accuracy per class. Just for a reference.

    Args:
        data: Data (numpy matrix) to train the model on.
        trueLabels: Labels (numpy array) just to calculate simple accuracy.
        model: Trained ML model.

    Returns:
        y_pred: Predicted labels (numpy array)
        y_probability_fin: Probability of predicted labels
        acc, accPerClass: Simple overall accuracy and accuracy per class  (just for fast check)
    '''

    # number of clases
    (unique_labels, counts) = np.unique(trueLabels, return_counts=True)
    numLabels = len(unique_labels)
    if (numLabels==1): #in specific case when in test set all the same label
        numLabels=2

    #PREDICT LABELS
    y_pred= model.predict(data)
    y_probability = model.predict_proba(data)

    #pick only probability of predicted class
    y_probability_fin=np.zeros(len(y_pred))
    indx=np.where(y_pred==1)
    if (len(indx[0])!=0):
        y_probability_fin[indx]=y_probability[indx,1]
    else:
        print('no seiz predicted')
    indx = np.where(y_pred == 0)
    if (len(indx[0])!=0):
        y_probability_fin[indx] = y_probability[indx,0]
    else:
        print('no non seiz predicted')

    #calculate accuracy
    diffLab=y_pred-trueLabels
    indx=np.where(diffLab==0)
    acc= len(indx[0])/len(trueLabels)

    # calculate performance and distances per class
    accPerClass=np.zeros(numLabels)
    for l in range(numLabels):
        indx=np.where(trueLabels==l)
        trueLabels_part=trueLabels[indx]
        predLab_part=y_pred[indx]
        diffLab = predLab_part - trueLabels_part
        indx2 = np.where(diffLab == 0)
        if (len(indx[0])==0):
            accPerClass[l] = np.nan
        else:
            accPerClass[l] = len(indx2[0]) / len(indx[0])

    return(y_pred, y_probability_fin, acc, accPerClass)




def kl_divergence(p,q):
    ''' Calculated Kulback Leibler divergence between two histograms
    '''

    delta=0.000001
    deltaArr=np.ones(len(p))*delta
    p=p+deltaArr
    q=q+deltaArr
    res=sum(p[i] * math.log2(p[i]/q[i]) for i in range(len(p)))
    return res

def js_divergence(p,q):
    ''' Calculated Jensen Shannon divergence between two histograms
    '''
    m=0.5* (p+q)
    res=0.5* kl_divergence(p,m) +0.5* kl_divergence(q,m)
    return (res)


def calcHistogramValues(sig, labels, histbins):
    ''' Calculates histogram of values  during seizure (label 1) and non seizure (label 0).

    Args:
        sig: Signal
        labels: Labels (0 is non-seizure and 1 is seizure)
        histbins: Number of bins of histogram.

    Returns: Histogram during seizure and histogram during non seizure

    '''
    numBins=int(histbins)
    sig2 = sig[~np.isnan(sig)]
    sig2 = sig2[np.isfinite(sig2)]

    # sig[sig == np.inf] = np.nan
    indxs=np.where(labels==0)[0]
    nonSeiz = sig[indxs]
    nonSeiz = nonSeiz[~np.isnan(nonSeiz)]
    try:
        nonSeiz_hist = np.histogram(nonSeiz, bins=numBins, range=(np.min(sig2), np.max(sig2)))
    except:
        print('Error with hist ')

    indxs = np.where(labels == 1)[0]
    Seiz = sig[indxs]
    Seiz = Seiz[~np.isnan(Seiz)]
    try:
        Seiz_hist = np.histogram(Seiz, bins=numBins, range=(np.min(sig2), np.max(sig2)))
    except:
        print('Error with hist ')

    # normalizing values that are in percentage of total samples - to not be dependand on number of samples
    nonSeiz_histNorm=[]
    nonSeiz_histNorm.append(nonSeiz_hist[0]/len(nonSeiz))
    nonSeiz_histNorm.append(nonSeiz_hist[1])
    Seiz_histNorm=[]
    Seiz_histNorm.append(Seiz_hist[0]/len(Seiz))
    Seiz_histNorm.append(Seiz_hist[1])
    return( Seiz_histNorm, nonSeiz_histNorm)


def calculateKLDivergenceForFeatures(dataset, patients, folderFeatures, TrueAnnotationsFile, FeaturesParams):
    ''' Funciton that loads data per patient from given dataset and calculate KL (JS) divergence per features of individual subjects.
    Also plots average over all subjects.

    Args:
        dataset: Dataset to use (important to load data properly)
        patients: Which subjects to analyse
        folderFeatures: Folder of input data features.
        TrueAnnotationsFile: True annotations file. Needed to know when is seizure and when is non-seizure.
        FeaturesParams: Class with various feature parameters.

    Returns: None. Saves output files and figures.
    '''
    numBins = 100
    folderFeaturesOut=folderFeatures+'/JSDivergence_'+'-'.join(FeaturesParams.featNames)+'/'
    os.makedirs(os.path.dirname(folderFeaturesOut), exist_ok=True)

    dataAllSubj = pd.DataFrame([])
    for patIndx, pat in enumerate(patients):
        dataOut = pd.DataFrame([])
        for fIndx, fName in enumerate(FeaturesParams.featNames):
            filesAll = np.sort(glob.glob(folderFeatures + '/' + pat + '/*' + fName + '.parquet.gzip'))
            print('-- Patient:', pat, 'NumSeizures:', len(filesAll))

            # load all files only once and mark where each file starts
            dataOut0 = concatenateDataFromFilesWithLabels(dataset, filesAll, TrueAnnotationsFile)  # with labels
            # concatenate for all features
            dataFixedCols = dataOut0[['Subject', 'FileName', 'Labels']]
            dataOut = pd.concat([dataOut, dataOut0.drop(['Subject', 'FileName', 'Time', 'Labels'], axis=1)], axis=1)

        # #rearange features so that first all features of ch1, then all features of ch2 etc
        # dataOutRear=pd.DataFrame([])
        # colNames=dataOut.columns.to_list()
        # for chIndx, ch in enumerate(DatasetPreprocessParams.channelNamesToKeep):
        #     cn = [s for s in colNames if ch in s]
        #     dataOutRear = pd.concat([dataOutRear, dataOut[cn]], axis=1)

        #concatenate rearenged features with subject, fileName, time and labels columns
        dataFinal=pd.concat([dataFixedCols, dataOut], axis=1)
        #concatenate for all subjects
        dataAllSubj=pd.concat([dataAllSubj, dataFinal], axis=0)

        # Calculate KL divergence for all features
        if (patIndx==0):
            allFeatNames = list(set(list(dataFinal.columns)) - set(['Subject', 'FileName', 'Labels']))
            allFeatNames.sort()
            JSdiverg=np.zeros((len(patients),len(allFeatNames)))
        for fi, feat in enumerate(allFeatNames):
            (SeizHist, nonSeizHist) = calcHistogramValues(dataFinal.loc[:,feat].to_numpy(), dataFinal.loc[:,'Labels'].to_numpy(),numBins)
            # KLdiverg_NSS[patIndx, f] = kl_divergence(nonSeizHist[0],SeizHist[0])
            JSdiverg[patIndx,fi] = js_divergence(nonSeizHist[0], SeizHist[0])

    #convert to DF
    JSdivergDF=pd.DataFrame(JSdiverg, columns=allFeatNames, index=patients)
    JSdivergDF.to_csv(folderFeaturesOut+'/JSdivergencePerSubjAndIndivFeature.csv', index=True)

    #group values per feature (all ch of the same feature together)
    JSdivergPerSubj_Mean=np.zeros((len(patients), len(FeaturesParams.allFeatNames)))
    JSdivergPerSubj_Std = np.zeros((len(patients), len(FeaturesParams.allFeatNames)))
    JSdivergTotal = np.zeros(( 2,len(FeaturesParams.allFeatNames)))
    for fi, feat in enumerate(FeaturesParams.allFeatNames):
        cols = [col for col in JSdivergDF.columns if feat in col]
        thisFeatData=JSdivergDF.loc[:,cols].to_numpy()
        JSdivergPerSubj_Mean[:,fi]=np.nanmean(thisFeatData,1)
        JSdivergPerSubj_Std[:,fi]=np.nanstd(thisFeatData,1)
        JSdivergTotal[0,fi]=np.nanmean(thisFeatData)
        JSdivergTotal[1,fi] = np.nanstd(thisFeatData)

    JSdivergPerSubj_MeanDF=pd.DataFrame(JSdivergPerSubj_Mean, columns=FeaturesParams.allFeatNames, index=patients)
    JSdivergPerSubj_StdDF=pd.DataFrame(JSdivergPerSubj_Std, columns=FeaturesParams.allFeatNames, index=patients)
    JSdivergTotalDF=pd.DataFrame(JSdivergTotal, columns=FeaturesParams.allFeatNames, index=['Mean','Std'])
    JSdivergPerSubj_MeanDF.to_csv(folderFeaturesOut+'/JSdivergencePerSubjAndFeature_Mean.csv', index=True)
    JSdivergPerSubj_StdDF.to_csv(folderFeaturesOut+'/JSdivergencePerSubjAndFeature_Std.csv', index=True)
    JSdivergTotalDF.to_csv(folderFeaturesOut+'/JSdivergencePerFeatures.csv', index=True)


    #PLOTTING
    # PLOTTING KL DIVERGENCE PER SUBJECT
    fig1 = plt.figure(figsize=(16, 16), constrained_layout=False)
    gs = GridSpec(6, 4, figure=fig1)
    fig1.subplots_adjust(wspace=0.4, hspace=0.6)
    xValues = np.arange(0,len(FeaturesParams.allFeatNames), 1)
    for p, pat in enumerate(patients):
        ax1 = fig1.add_subplot(gs[int(np.floor(p / 4)), np.mod(p, 4)])
        ax1.errorbar(xValues, JSdivergPerSubj_Mean[p, :], yerr=JSdivergPerSubj_Std[p, :], fmt='b', label='JS')
        ax1.legend()
        ax1.set_xlabel('Feature')
        ax1.set_ylabel('Divergence')
        ax1.set_title('Subj ' + pat)
        ax1.grid()
    fig1.show()
    fig1.savefig(folderFeaturesOut + '/JSDivergence_PerSubj.png', bbox_inches='tight')
    plt.close(fig1)

    #Plot average of all subj
    fig1 = plt.figure(figsize=(16, 16), constrained_layout=False)
    gs = GridSpec(1, 1, figure=fig1)
    fig1.subplots_adjust(wspace=0.4, hspace=0.6)
    xValues = np.arange(0,len(FeaturesParams.allFeatNames), 1)
    ax1 = fig1.add_subplot(gs[0,0])
    ax1.errorbar(xValues, JSdivergTotal[0, :], yerr=JSdivergTotal[1, :], fmt='b', label='JS')
    ax1.legend()
    ax1.set_xlabel('Feature')
    ax1.set_ylabel('Divergence')
    ax1.set_xticks(np.arange(0,len(FeaturesParams.allFeatNames), 1))
    ax1.set_xticklabels(FeaturesParams.allFeatNames, fontsize=10, rotation=45, ha='right', rotation_mode='anchor')
    ax1.set_title('AllSubj')
    ax1.grid()
    fig1.show()
    fig1.savefig(folderFeaturesOut + '/JSDivergence_AllSubj.png', bbox_inches='tight')
    plt.close(fig1)


def plotPredictionsMatchingInTime(trueLabels, predLabels, predLabels_MovAvrg, predLabels_Bayes,outName, PerformanceParams):
    ''' Plots predictions and true labels in time. Plots for raw predictions and two types of postprocessing (moving average and Bayes).
    Args:
        trueLabels: True labels array
        predLabels: Predicted raw labels
        predLabels_MovAvrg: Predicted labels after moving average postprocessing
        predLabels_Bayes: Predicted labels after Bayes postprocessing
        outName: Output file name
        PerformanceParams: Class with various performance parameters
    '''
    param = scoring.EventScoring.Parameters(
        toleranceStart=PerformanceParams.toleranceStart,
        toleranceEnd=PerformanceParams.toleranceEnd,
        minOverlap=PerformanceParams.minOveralp,
        maxEventDuration=PerformanceParams.maxEventDuration,
        minDurationBetweenEvents=PerformanceParams.minDurationBetweenEvents)
    ref = Annotation(trueLabels, PerformanceParams.predictionFreq)

    fig, axs = plt.subplots(nrows=3, ncols=2, figsize=(12, 6) )

    hyp = Annotation(predLabels,  PerformanceParams.predictionFreq)
    visualization.plotSampleScoring(ref, hyp, ax=axs[0,0])
    axs[0,0].set_title('Sample based matching')
    axs[0, 0].set_ylabel('Raw predictions')
    visualization.plotEventScoring(ref, hyp, param, ax=axs[0,1])
    axs[0,1].set_title('Event based matching')

    hyp = Annotation(predLabels_MovAvrg, PerformanceParams.predictionFreq)
    visualization.plotSampleScoring(ref, hyp, ax=axs[1,0])
    axs[1, 0].set_ylabel('MovAvrg Postprocessing')
    visualization.plotEventScoring(ref, hyp, param, ax=axs[1,1])

    hyp = Annotation(predLabels_Bayes, PerformanceParams.predictionFreq)
    visualization.plotSampleScoring(ref, hyp, ax=axs[2,0])
    axs[2, 0].set_ylabel('Bayes Postprocessing')
    visualization.plotEventScoring(ref, hyp, param, ax=axs[2,1])

    fig.savefig(outName+'.png', bbox_inches='tight')
    # fig.savefig(outName+'.svg', bbox_inches='tight')
    plt.close(fig)


def loadAllSubjData(dataset, inputFolder, patients, featNames,channelNamesToKeep, TrueAnnotationsFile):
    ''' Loads all files (calculated features) belonging to one listed subjects

    Args:
        dataset: Dataset which is used, to account for differences in file names
        inputFolder: Folder with calculated features that will be read
        patients: All subject whose files we want to load
        featNames: List of features to keep
        channelNamesToKeep: List of channels to keep
        TrueAnnotationsFile: Annotations dataframe that contains labels

    Returns: Dataframe containing all features of given subjects.
    Colums of dataframe are: 'Subject', 'FileName', 'Time', 'Labels', all features...
    '''
    dataAllSubj = pd.DataFrame([])
    for patIndx, pat in enumerate(patients):
        dataFinal=loadOneSubjData(dataset, pat, inputFolder, featNames,channelNamesToKeep, TrueAnnotationsFile)
        # concatenate for all subjects
        dataAllSubj = pd.concat([dataAllSubj, dataFinal], axis=0)

    # replace nan values with mean of columns
    dataAllSubj = removeExtremeValues(dataAllSubj)
    return (dataAllSubj)

def loadOneSubjData(dataset, pat, inputFolder, featNames,channelNamesToKeep, TrueAnnotationsFile):
    ''' Loads all files (calculated features) belonging to one subject

    Args:
        dataset: Dataset which is used, to account for differences in file names
        pat: Subject whose files to load
        inputFolder: Folder with calculated features that will be read
        featNames: List of features to keep
        channelNamesToKeep: List of channels to keep
        TrueAnnotationsFile: Annotations dataframe that contains labels

    Returns: Dataframe containing all features of given subject.
     Colums of dataframe are: 'Subject', 'FileName', 'Time', 'Labels', all features...
    '''
    dataOut = pd.DataFrame([])
    for fIndx, fName in enumerate(featNames):
        filesAll = np.sort(glob.glob(inputFolder + '/' + pat + '/*' + fName + '.parquet.gzip'))
        print('-- Patient:', pat, 'NumSeizures:', len(filesAll))
        # load all files only once and mark where each file starts
        # (dataOut0, startIndxOfSubjects) = concatenateDataFromFiles(filesAll) # without labels
        dataOut0 = concatenateDataFromFilesWithLabels(dataset, filesAll, TrueAnnotationsFile)  # with labels
        # concatenate for all features
        dataFixedCols = dataOut0[['Subject', 'FileName', 'Time', 'Labels']]
        dataOut = pd.concat([dataOut, dataOut0.drop(['Subject', 'FileName', 'Time', 'Labels'], axis=1)], axis=1)

    # rearange features so that first all features of ch1, then all features of ch2 etc
    dataOutRear = pd.DataFrame([])
    colNames = dataOut.columns.to_list()
    for chIndx, ch in enumerate(channelNamesToKeep):
        cn = [s for s in colNames if ch in s]
        dataOutRear = pd.concat([dataOutRear, dataOut[cn]], axis=1)

    # concatenate rearenged features with subject, fileName, time and labels columns
    dataFinal = pd.concat([dataFixedCols, dataOutRear], axis=1)
    dataFinal=dataFinal.reset_index(drop=True)
    return dataFinal

def findMinNumHoursToTrain(data, minHours, stepInHours):
    ''' Return minimum amount of data (in hours) with which to start train. It is minHours or at least one seizure.

    Args:
        data: Dataframe containing (feature) data with labels too
        minHours: How much minimally hours we need at the beginning to train
        stepInHours: How much new data will be added in each CV (in hours)
    Returns: minHours if there is at least one seizure within it, otherwise min full hour with at least one seizure
    '''
    labels=data['Labels'].to_numpy()
    startIndx = np.where(np.diff(labels) == 1)[0] + 1
    endIndx = np.where(np.diff(labels) == -1)[0] + 1
    if (endIndx[0]< minHours*60*60): # seizure within first minHours of recordings
        minHoursOut=int(minHours/stepInHours)
    else:
        minHoursOut=int(np.ceil(endIndx[0]/(stepInHours*60*60)))
    return minHoursOut

# def movingAvrgSmoothing(prediction,  winLen, thrPerc):
#     ''' Returns labels after  moving average postprocessin - if more than thrPerc of labels  in wilLen long window are 1 final label is 1 otherwise 0.
#     Args:
#         prediction:  Predictions (numpy array)
#         winLen: Window length in with which moving average is performed.
#         thrPerc: Percentage of labels in the window that has to be 1 for final label to be 1.
#     Returns: Postprocessed predictions
#     '''
#
#     smoothLabelsStep1=np.zeros((len(prediction)))
#     for i in range(int(winLen), int(len(prediction))):
#         s= sum( prediction[i-winLen+1: i+1] )/winLen
#         if (s>= thrPerc):  #and prediction[i]==1
#             smoothLabelsStep1[i]=1
#     return  (smoothLabelsStep1)


def movingAvrgSmoothing(data,  winLen, thrPerc):
    ''' Returns labels after  moving average postprocessin - if more than thrPerc of labels  in wilLen long window are 1 final label is 1 otherwise 0.
    Args:
        prediction:  Predictions (numpy array)
        winLen: Window length in with which moving average is performed.
        thrPerc: Percentage of labels in the window that has to be 1 for final label to be 1.
    Returns: Postprocessed predictions
    '''
    kernel = np.ones(winLen) / winLen
    data_convolved = np.convolve(data, kernel, mode='same') #calculating averga
    dataOut=np.zeros(len(data))
    dataOut[data_convolved>=thrPerc]=1
    return(dataOut)


def mergeTooCloseSeizures(data, minDist):
    ''' Returns labels after  merging seizure groups that are closer than minDist.
    Args:
        data:  Predictions (numpy array)
        minDist: If two seizures are close than minDist samples then everything in between is converted to seizures too (to 1s).
    Returns: Postprocessed predictions
    '''
    startIndx = np.where(np.diff(data) == 1)[0] + 1
    endIndx = np.where(np.diff(data) == -1)[0] + 1
    dataOut=np.copy(data)
    for i in range(0,len(startIndx)-1):
        if (startIndx[i+1]-endIndx[i])<minDist:
            dataOut[endIndx[i]:startIndx[i+1]]=1
    return dataOut

def smoothenLabels_Bayes(prediction,  probability, winLen,  probThresh):
    ''' Returns labels after Bayes postprocessing:
    calculates cummulative probability of seizure and non seizure over the window of size winLen
    and if log (cong_pos /cong_ned )> probThresh then  it should be seizrue (label 1).

    Args:
        prediction: Predictions (numpy array)
        probability: Prediction probabilities (numpy array)
        winLen: Window length in with which postprocessing is performed.
        probThresh: Log threshold which if exceeded it is high probability enough to be seizure (label 1)
    Returns: Postprocessed predictions
    '''

    #convert probability to probability of pos
    probability_pos=np.copy(probability)
    indxs=np.where(prediction==0)[0]
    probability_pos[indxs]=1-probability[indxs]

    smoothLabels=np.zeros((len(prediction)))
    # confAll=np.zeros(len(prediction))
    for i in range(int(winLen), int(len(prediction))):
        probThisWind=probability_pos[i-winLen+1: i+1]
        conf_pos=np.prod(probThisWind)
        conf_neg=np.prod( 1-probThisWind)
        conf=np.log( (conf_pos+ 0.00000001) /(conf_neg + 0.00000001))
        if (conf>= probThresh):  #and prediction[i]==1
                smoothLabels[i]=1
    return  (smoothLabels)


def performance_sampleAndEventBased(predLab, trueLab, PerformanceParams):
    ''' Function that returns 9 different performance measures of prediction on epilepsy
    - on the level of seizure events/episodes (sensitivity, precision and F1 score)
    - on the level of seizure samples, or each sample (sens, prec, F1)
    - combination of F1 scores for events and samples ( mean or gmean)
    - number of false positives per day (false alarm rate FAR)

    Args:
        predLab: Predicted labels (numpy array)
        trueLab: True labels (numpy array)
        PerformanceParams: Various performance parameters in PerformanceParams class

    Returns: 9 values: E_sens, E_prec, E_f1score, S_sens, S_prec, S_f1score, ES_mean, ES_gmean, FAR
    '''
    ref = Annotation(trueLab, PerformanceParams.predictionFreq)
    hyp = Annotation(predLab, PerformanceParams.predictionFreq)

    performanceSamples = scoring.SampleScoring(ref, hyp)
    # figSamples = visualization.plotSampleScoring(ref, hyp)

    param = scoring.EventScoring.Parameters(
        toleranceStart=PerformanceParams.toleranceStart,
        toleranceEnd=PerformanceParams.toleranceEnd,
        minOverlap=PerformanceParams.minOveralp,
        maxEventDuration=PerformanceParams.maxEventDuration,
        minDurationBetweenEvents=PerformanceParams.minDurationBetweenEvents)
    performanceEvents = scoring.EventScoring(ref, hyp, param)
    # figEvents = visualization.plotEventScoring(ref, hyp, param)

    #calculate combinations
    F1DEmean=(performanceEvents.f1+performanceSamples.f1)/2
    F1DEgeoMean=np.sqrt(performanceEvents.f1*performanceSamples.f1)

    return(  performanceEvents.sensitivity, performanceEvents.precision, performanceEvents.f1, performanceSamples.sensitivity, performanceSamples.precision, performanceSamples.f1, F1DEmean, F1DEgeoMean, performanceEvents.fpRate)

# def TestDifferentPostprocessingParams(outPredictionsFolder, dataset, GeneralParams, StandardMLParams):
#     AllSubjLabels = pd.DataFrame()
#     for patIndx, pat in enumerate(GeneralParams.patients):
#         print(pat)
#         InName = outPredictionsFolder + '/Subj' + pat + '_' + StandardMLParams.modelType + '_TestPredictions.csv.parquet.gzip'
#         data = readDataFromFile(InName)
#         SubjNames = pd.DataFrame([pat] * data.shape[0], columns=['Subject'])
#         LabelsToKeep = pd.concat([SubjNames, data.loc[:, ['TrueLabels', 'ProbabLabels', 'PredLabels']]], axis=1)
#         AllSubjLabels = pd.concat([AllSubjLabels, LabelsToKeep], axis=0)
#     # for different params setups measure
#     winLenArr = [5, 10, 15, 30]
#     thrArr = [0.25, 0.5, 0.75, 1]
#     mergeArr = [0, 2, 5, 10, 30]
#     MovingAverageResults = pd.DataFrame()
#     for w in winLenArr:
#         for thr in thrArr:
#             for m in mergeArr:
#                 smtType = 'Win' + str(w) + '_Thr' + str(thr) + '_Merge' + str(m)
#                 # run for each subject
#                 perfAllSubj = np.zeros((len(GeneralParams.patients), 9))
#                 for patIndx, pat in enumerate(GeneralParams.patients):
#                     thisSubjData = AllSubjLabels[AllSubjLabels['Subject'] == pat]
#                     # postprocess labels
#                     lab = movingAvrgSmoothing(thisSubjData['PredLabels'].to_numpy(), w, thr)
#                     lab = mergeTooCloseSeizures(lab, m)
#                     perf = performance_sampleAndEventBased(lab, thisSubjData['TrueLabels'].to_numpy(),
#                                                            PerformanceParams)
#                     perfAllSubj[patIndx, :] = np.asarray(perf).reshape((1, -1))
#                     perfDF = pd.DataFrame(perfAllSubj[patIndx, :].reshape((1, -1)),
#                                           columns=['SensE', 'PrecE', 'F1E', 'SensS', 'PrecS', 'F1S', 'F1mean',
#                                                    'F1gmean', 'FPrate'])
#                     perfDF.insert(loc=0, column='Subject', value=pat)
#                     perfDF.insert(loc=0, column='PostprocessType', value=smtType)
#                     MovingAverageResults = pd.concat([MovingAverageResults, perfDF], axis=0)
#                 # calcualte avrg for all subj
#                 perfDF = pd.DataFrame(np.nanmean(perfAllSubj, 0).reshape((1, -1)),
#                                       columns=['SensE', 'PrecE', 'F1E', 'SensS', 'PrecS', 'F1S', 'F1mean', 'F1gmean',
#                                                'FPrate'])
#                 perfDF.insert(loc=0, column='Subject', value='AllSubjAvrg')
#                 perfDF.insert(loc=0, column='PostprocessType', value=smtType)
#                 MovingAverageResults = pd.concat([MovingAverageResults, perfDF], axis=0)
#     # for different params setups measure
#     winLenArr = [5, 10, 30]
#     thrArr = [0, 0.5, 1]
#     mergeArr = [0, 10, 30]
#     BayesResults = pd.DataFrame()
#     for w in winLenArr:
#         for thr in thrArr:
#             for m in mergeArr:
#                 smtType = 'Win' + str(w) + '_Thr' + str(thr) + '_Merge' + str(m)
#                 # run for each subject
#                 perfAllSubj = np.zeros((len(GeneralParams.patients), 9))
#                 for patIndx, pat in enumerate(GeneralParams.patients):
#                     thisSubjData = AllSubjLabels[AllSubjLabels['Subject'] == pat]
#                     # postprocess labels
#                     lab = smoothenLabels_Bayes(thisSubjData['PredLabels'].to_numpy(),
#                                                thisSubjData['ProbabLabels'].to_numpy(), w, thr)
#                     lab = mergeTooCloseSeizures(lab, m)
#                     perf = performance_sampleAndEventBased(lab, thisSubjData['TrueLabels'].to_numpy(),
#                                                            PerformanceParams)
#                     perfAllSubj[patIndx, :] = np.asarray(perf).reshape((1, -1))
#                     perfDF = pd.DataFrame(perfAllSubj[patIndx, :].reshape((1, -1)),
#                                           columns=['SensE', 'PrecE', 'F1E', 'SensS', 'PrecS', 'F1S', 'F1mean',
#                                                    'F1gmean', 'FPrate'])
#                     perfDF.insert(loc=0, column='Subject', value=pat)
#                     perfDF.insert(loc=0, column='PostprocessType', value=smtType)
#                     BayesResults = pd.concat([BayesResults, perfDF], axis=0)
#                 # calcualte avrg for all subj
#                 perfDF = pd.DataFrame(np.nanmean(perfAllSubj, 0).reshape((1, -1)),
#                                       columns=['SensE', 'PrecE', 'F1E', 'SensS', 'PrecS', 'F1S', 'F1mean', 'F1gmean',
#                                                'FPrate'])
#                 perfDF.insert(loc=0, column='Subject', value='AllSubjAvrg')
#                 perfDF.insert(loc=0, column='PostprocessType', value=smtType)
#                 BayesResults = pd.concat([BayesResults, perfDF], axis=0)
#
#     print('Finding the best')
#     AvrgsAllSubj = MovingAverageResults[MovingAverageResults['Subject'] == 'AllSubjAvrg']
#     AvrgsAllSubj = AvrgsAllSubj.reset_index(drop=True)
#     AvrgsAllSubj['F1gmean'].idxmax()
#
#     print('Finding the best')
#     BayesAllSubj = BayesResults[BayesResults['Subject'] == 'AllSubjAvrg']
#     BayesAllSubj = BayesAllSubj.reset_index(drop=True)
#     BayesAllSubj['F1gmean'].idxmax()
#
#     # save performance per file
#     OutputName = outPredictionsFolder + '/' + dataset + '_OptimizingPostprocessParameters_MovAvrg.csv'
#     MovingAverageResults.to_csv(OutputName, index=False)
#     OutputName = outPredictionsFolder + '/' + dataset + '_OptimizingPostprocessParameters_Bayes.csv'
#     BayesResults.to_csv(OutputName, index=False)
#     OutputName = outPredictionsFolder + '/' + dataset + '_OptimizingPostprocessParameters_MovAvrg_SubjAvrg.csv'
#     AvrgsAllSubj.to_csv(OutputName, index=False)
#     OutputName = outPredictionsFolder + '/' + dataset + '_OptimizingPostprocessParameters_Bayes_SubjAvrg.csv'
#     BayesAllSubj.to_csv(OutputName, index=False)