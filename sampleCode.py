#Libraries
import numpy as np
from scipy import stats
from keras import models
from keras.models import load_model
from keras import backend as K

#Sets the default float type
K.set_floatx('float64')

#Change here
################################################################################
#Input file directory
filenameForWctBar = 'filteredReactionRate.raw'
filenameForNctBar = 'filteredScalarGradient.raw'

#Output file directory
filenameForPhiBar = 'filteredVolumeFractionOfRXN.raw'

#Load trained NN model (.h5 file)
nnModel = load_model('nnModel.h5')

#Filter size in thermal thickness of a laminar flame (value between 0.50 and 2.00)
filterSizeInDth = 1.0

#Laminar flame characteristics
du      = 0.36408879999999999       #Unburnt mixture density [kg/m^3]
sl      = 10.400000000000000        #Laminar flame speed [m/s]   
deltath = 4.90000005811452866E-004  #Thermal thickness of a laminar flame (temperature gradient based) [m]
################################################################################

#Normalization constants
normConst_wct = du * sl / deltath  
normConst_nct = 1.0 / deltath
normConst_flt = 2.0

#Check
print('normConst_wct:',normConst_wct)
print('normConst_nct:',normConst_nct)
print('normConst_flt:',normConst_flt)
print('')

#Loading wctBar and nctBar
wctBarPlus     = np.fromfile(filenameForWctBar, count=-1, dtype=np.float64)
wctBarPlus     = wctBarPlus.reshape(np.shape(wctBarPlus)[0],1)
nctNormBarPlus = np.fromfile(filenameForNctBar, count=-1, dtype=np.float64)
nctNormBarPlus = nctNormBarPlus.reshape(np.shape(nctNormBarPlus)[0],1)

#Make array for filterSizePlus
filterSizePlus = np.zeros((np.shape(wctBarPlus)[0],1))

#Normalizing the data (making wctBarPlus, nctNormBarPlus and filterSizePlus)
wctBarPlus          = wctBarPlus      / normConst_wct
nctNormBarPlus      = nctNormBarPlus  / normConst_nct
filterSizePlus[:,0] = filterSizeInDth / normConst_flt

#Check 
print('wctBarPlus_shape:'    ,wctBarPlus.shape)
print('wctBarPlus_stat:'     ,stats.describe(wctBarPlus))
print('nctNormBarPlus_shape:',nctNormBarPlus.shape)
print('nctNormBarPlus_stat:' ,stats.describe(nctNormBarPlus))
print('filterSizePlus_shape:',filterSizePlus.shape)
print('filterSizePlus_stat:' ,stats.describe(filterSizePlus))

#Concatenating wctBarPlus, nctNormBarPlus and filterSizePlus into X
X = np.concatenate((wctBarPlus, nctNormBarPlus, filterSizePlus), axis=1)

#Prediction 
y = np.zeros((np.shape(X)[0]))
y = nnModel.predict(X, batch_size=16, verbose=2)

#Check 
print('X_shape:',X.shape)
print('X_stat:' ,stats.describe(X))
print('y_shape:',y.shape)
print('y_stat:' ,stats.describe(y))
print('')

#Prediction result into a file
fileobj  = open(filenameForPhiBar, mode='wb')
y.tofile(fileobj)
fileobj.close



