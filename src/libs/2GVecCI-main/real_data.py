#--------------- IMPORTS -----------------
import numpy as np
import matplotlib
import netCDF4
from netCDF4 import Dataset,num2date
import datetime
import pandas as pd
from matplotlib import pyplot as plt
import xarray as xr
import Functions_2GVecCI as fcs
from tigramite import data_processing as pp
#------------------------------------------

xdata = xr.open_dataset('AirTempData.nc')
crit_list = []


for i in range(2,5): # grid coarsening parameter for NINO longitude
    for k in range(1,4): # grid coarsening parameter NINO latitude, smaller range because NINo 3.4 has limited latitudinal grid-boxes 
        for j in range(2,5): # grid coarsening parameter for BCT latitude
            for l in range(2,5): # grid coarsening parameter for BCT longitude
                print(k,i,j,l)
                
                #ENSO LAT 6,-6, LON 190, 240
                #BCT LAT 65,50 LON 200, 240
                #TATL LAT 25, 5, LON 305, 325

                Xregion=xdata.sel(lat=slice(6.,-6.,k), lon = slice(190.,240.,i))
                Yregion=xdata.sel(lat=slice(65.,50.,j), lon = slice(200.,240.,l))

                
                # de-seasonlize
                #----------------
                monthlymean = Xregion.groupby("time.month").mean("time")

                anomalies_Xregion = Xregion.groupby("time.month") - monthlymean

                Yregion_monthlymean = Yregion.groupby("time.month").mean("time")

                anomalies_Yregion = Yregion.groupby("time.month") - Yregion_monthlymean


                # functions to consider triples on months
                #-----------------------------------------

                def is_ond(month):
                    return (month >= 10) & (month <= 12)

                def is_son(month):
                    return (month >= 9) & (month <= 11)

                def is_ndj(month):
                    return ((month >= 11) & (month <= 12)) or (month==1)

                def is_jfm(month):
                    return (month >= 1) & (month <= 3)

                # NINO for oct-nov-dec
                #--------------------

                ond_Xregion = anomalies_Xregion.sel(time=is_ond(xdata['time.month']))

                ond_Xregion_by_year = ond_Xregion.groupby("time.year").mean()

                num_ond_Xregion = np.array(ond_Xregion_by_year.to_array())[0]

                reshaped_Xregion = np.reshape(num_ond_Xregion, newshape = (num_ond_Xregion.shape[0],num_ond_Xregion.shape[1]*num_ond_Xregion.shape[2]))

                #BCT for jan-feb-mar
                #-------------------

                jfm_Yregion = anomalies_Yregion.sel(time=is_jfm(xdata['time.month']))

                jfm_Yregion_by_year = jfm_Yregion.groupby("time.year").mean()

                num_jfm_Yregion = np.array(jfm_Yregion_by_year.to_array())[0]

                reshaped_Yregion = np.reshape(num_jfm_Yregion, newshape = (num_jfm_Yregion.shape[0],num_jfm_Yregion.shape[1]*num_jfm_Yregion.shape[2]))

                #Consider cases where group sizes are not further apart than 10 grid boxes
                #------------------------------------------------------------------------
                if abs(reshaped_Xregion.shape[1]-reshaped_Yregion.shape[1])<10:

                    #GAUSSIAN KERNEL SMOOTHING
                    #-------------------------
                    for var in range(reshaped_Xregion.shape[1]):
                         reshaped_Xregion[:, var] = pp.smooth(reshaped_Xregion[:, var], smooth_width=12 * 10, kernel='gaussian', mask=None,
                                                          residuals=True)
                    for var in range(reshaped_Yregion.shape[1]):
                         reshaped_Yregion[:, var] = pp.smooth(reshaped_Yregion[:, var], smooth_width=12 * 10, kernel='gaussian', mask=None,
                                                         residuals=True)
                    #######
                    def shift_by_one(array1, array2, t):
                        if t == 0:
                            return array1, array2
                        elif t < 0:
                            s = -t
                            newarray1 = array1[:-s, :]
                            newarray2 = array2[s:, :]
                            return newarray1, newarray2

                        else:
                            newarray1 = array1[t:, :]
                            newarray2 = array2
                            return newarray1, newarray2

                    shifted_Yregion, shifted_Xregion = shift_by_one(reshaped_Yregion,reshaped_Xregion,1)

                    tigra_Xregion = pp.DataFrame(shifted_Xregion)

                    tigra_Yregion = pp.DataFrame(shifted_Yregion)

                    print(reshaped_Xregion.shape, reshaped_Yregion.shape)
                    print(shifted_Xregion.shape, shifted_Yregion.shape)

                    # CHANGE TEST TO 'PC' FOR 2G-VecCi.PC and 'full' FOR 2g-VecCi.Full
                    #------------------------------------------------------------------

                    results = fcs.identify_causal_direction(tigra_Xregion,tigra_Yregion,alpha=0.01, type = 'both',CI_test_method='ParCorr',ambiguity = 0.01, test = 'full', max_sep_set = None, linear = True)
                    crit = results[2]- results[1]
                    print(crit, results)
                    print(fcs.identify_causal_direction_trace_method(tigra_Xregion, tigra_Yregion) )
                    crit_list.append(crit)

print(crit_list)

fig = plt.hist(crit_list,bins=20)
filename = 'HIST_nino_bct_Full_smoothed' # OR 'HIST_nino_bct_PC_smoothed' resp.
plt.savefig(filename, bbox_inches="tight")
plt.close()

#######################################
#######################################

print((np.array(crit_list)).mean()) # mean 
print((np.array(crit_list)).std())  # standard deviation
print((np.array(crit_list)>0.01).sum()) # fraction of correct inferences when alpha = 0.01
print((np.array(crit_list)<-0.01).sum()) # fraction of wrong inferences when alpha = 0.01

