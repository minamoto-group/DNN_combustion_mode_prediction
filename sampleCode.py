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
wct_bar = 'filtered_reaction_rate.raw'
nct_bar = 'filtered_scalar_gradient.raw'

#Filtersize in thermal thickness of a laminar flame (value between 0.50 and 2.00)
filtersize_dth = 1.0

#Load trained NN model (.h5 file)
nn_model = load_model('nn_model.h5')

#Output file directory
phi_bar = 'filtered_volume_fraction_of_local_combustion_mode.raw'

#Laminar flame characteristics
du = 0.3641     #Unburnt mixture density [kg/m^3]
sl = 10.4000    #Laminar flame speed [m/s]   
dth = 0.00049   #Thermal thickness of a laminar flame (temperature gradient based) [m]
################################################################################

#Normalization constants
CONSTANT_WCT = du * sl / dth  
CONSTANT_NCT = 1.0 / dth
CONSTANT_FILTERSIZE = 2.0

#Check
print('CONSTANT_WCT:', CONSTANT_WCT)
print('CONSTANT_NCT:', CONSTANT_NCT)
print('CONSTANT_FILTERSIZE:', CONSTANT_FILTERSIZE)
print('')

#Loading wct_bar and nct_bar
wct_bar_plus = np.fromfile(wct_bar, count=-1, dtype=np.float64)
wct_bar_plus = wct_bar_plus.reshape(np.shape(wct_bar_plus)[0],1)
nct_bar_plus = np.fromfile(nct_bar, count=-1, dtype=np.float64)
nct_bar_plus = nct_bar_plus.reshape(np.shape(nct_bar_plus)[0],1)

#Make array for filtersize_dth_plus
filtersize_dth_plus = np.zeros((np.shape(wct_bar_plus)[0],1))

#Normalizing the data (making wct_bar_plus, nct_bar_plus and filtersize_dth_plus)
wct_bar_plus = wct_bar_plus / CONSTANT_WCT
nct_bar_plus = nct_bar_plus / CONSTANT_NCT
filtersize_dth_plus[:,0] = filtersize_dth / CONSTANT_FILTERSIZE

#Check 
print('wct_bar_plus_shape:', wct_bar_plus.shape)
print('wct_bar_plus_stat:', stats.describe(wct_bar_plus))
print('nct_bar_plus_shape:', nct_bar_plus.shape)
print('nct_bar_plus_stat:', stats.describe(nct_bar_plus))
print('filtersize_dth_plus_shape:', filtersize_dth_plus.shape)
print('filtersize_dth_plus_stat:', stats.describe(filtersize_dth_plus))

#Concatenating wct_bar_plus, nct_bar_plus and filtersize_dth_plus into nn_input
nn_input = np.concatenate((wct_bar_plus, nct_bar_plus, filtersize_dth_plus), axis=1)

#Prediction 
nn_output = np.zeros((np.shape(nn_input)[0]))
nn_output = nn_model.predict(nn_input, batch_size=16, verbose=2)

#Check 
print('nn_input_shape:', nn_input.shape)
print('nn_input_stat:', stats.describe(nn_input))
print('nn_output_shape:', nn_output.shape)
print('nn_output_stat:', stats.describe(nn_output))
print('')

#Prediction result into a file
with open(phi_bar, mode='wb') as f:
    f.write(nn_output)