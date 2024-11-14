import numpy as np
import matplotlib.pyplot as plt
import dedalus.public as d3
import pathlib
import subprocess
import h5py
from dedalus.extras.plot_tools import plot_bot_2d
figkw = {'figsize':(6,4), 'dpi':100}

fig = plt.figure(figsize=(6, 4), dpi=100)
plt.xlabel('simulation time (free falls)')
plt.ylabel('Reynolds Number')
plt.title("Changes in Reynold's number over time, Ra=1.5e5")
plt.legend()

with h5py.File("analysis/Ra=1e7_time=200/analysis_s1_p0.h5", mode='r') as file:
    Re = file['tasks']['Re']
    #t = Re.dims[0]['sim_time']
    #plt.plot(t[:],(Re)[:,0], label='Ra=1e5')
    print(list(file['tasks']))
    print(list(file['tasks']['Re']))

inputfile = "analysis/Ra=1.7075e3_time=2000/analysis_s1.h5"
versions = [3,3]
with h5py.File(inputfile, mode='r') as file:
    allscales = list(file['scales'])
    alltasks = list(file['tasks'])
    print("Scales available in this file are ", allscales)
    print("Tasks available in this file are ", alltasks)        
    times = np.array(file['scales']['sim_time'])
    iterations = np.array(file['scales']['iteration'])
    z = np.array(file['scales']['z_hash_2b3e6c1ad6197e7bbb577c37c9be3babe1727daf'])[:]

    bx_all = np.array(file['tasks']['<b>_x'])[:,0,:]
    
    L_cond_all_tot = np.array(file['tasks']['L_cond_tot'])[:,0,:]


    L_conv_all = np.array(file['tasks']['L_conv'])[:,0,:]

    Re_all = np.array(file['tasks']['Re'])[:,0,0]
    Nu_all = np.array(file['tasks']['Nusselt'])[:,0,0]   
    wT_all = np.array(file['tasks']['wT'])[:,0,0]  
    visc_dissip_all = np.array(file['tasks']['visc_dissip'])[:,0,0]  
    visc_dissip_all = visc_dissip_all[:,0,0]

    visc_dissip_vort_all = np.array(file['tasks']['visc_dissip_vort'])[:,0,0]  
    gradusq_all = np.array(file['tasks']['<(grad u)^2>'])[:,0,0]  

def find_nearest_index(array_of_times, target_time):
    #helper function to find the index of element in array_of_times that is closest to target_time
    
    index = (np.abs(array_of_times - target_time)).argmin()
    return index



averaging_method = 'numpy mean'
def get_timeavg(quantity_array, timearray, start_time, end_time):
    if (start_time < timearray[0]):
        print("Requested starting time is too early, beginning average at start of data")
        start_time = timearray[0]
    if (end_time> timearray[-1]):
        print("Requested ending time is too late, ending average at end of data")
        end_time = timearray[-1]
    
    start_index = find_nearest_index(timearray, start_time)
    end_index = find_nearest_index(timearray, end_time)
    
    if (averaging_method== 'numpy mean'):
        mean_quantity = np.mean(quantity_array[start_index:end_index], axis=0)
    else:
        print("Only numpy mean averaging supported!")
        
    return mean_quantity

#look at time-averaged heat fluxes for a particular case

Rayleigh=1.0e5
Prandtl=1.0
kappa = (Rayleigh*Prandtl)**(-1/2)
print(kappa)
casenum = 1

averaging_interval = 50 # number of free-fall times over which to average
end_time = times[-1]
start_time = end_time - averaging_interval



L_conv_average = get_timeavg(L_conv_all, times, start_time, end_time)
L_cond_average = get_timeavg(L_cond_all_tot, times, start_time, end_time)
L_tot_average = L_conv_average+ L_cond_average



fig, ax=plt.subplots()


#ax.plot(L_conv_average, z, label=r'$L_{ conv}$')

ax.plot(L_cond_average, z, label=r'$L_{ cond}$')

#ax.plot(L_tot_average, z, label=r'$L_{ tot}$')

ax.set_title('Conductive and Convective luminosities, Ra = 1e5')

ax.set_xlabel('Energy flux')

ax.set_ylabel('z')

ax.legend()

plt.savefig('analysis/Conductive_Luminosity_Ra=1e5')
#plt.savefig('analysis/Convective_Luminosity_Ra=1707,5')
#plt.savefig('analysis/Heat_Transfer_Ra=1707,5')

#compare mean temperature profiles in all cases

fig, ax=plt.subplots()

num_cases = len(times)
num_freefalltimes_for_average = 50
add_background = False# do we need to add in background state?  Only do this if solved for perturbation
for casenum in range(num_cases):
    # get bx (horizontally averaged T perturbation) for this case, along with z and t arrays
    
    #get time-averaged bx over last N free-fall times of run
    
    averaging_interval = num_freefalltimes_for_average 
    end_time = times[-1]
    start_time = end_time - averaging_interval
    bx_average = get_timeavg(bx_all, times, start_time, end_time)


    if (add_background): 
        # add in the background temperature profile -- remember we are solving for perturbation
        base_state_Tprofile = 1.0 - z
    
        bx_average_withbasestate = bx_average + base_state_Tprofile
    else:
        bx_average_withbasestate = bx_average
    this_label = 'case '+str(casenum)
    ax.plot(z,bx_average_withbasestate, label=this_label)

ax.set_xlabel(r'$T(z)$')
ax.set_ylabel(r'$z$')
ax.set_title('Temperature Profile between plates')
plt.savefig('analysis/TemperatureProfile_Ra=1707,5')
