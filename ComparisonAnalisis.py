#base code provided by Matthew Browning, modified to suit our data. 

#compare cases at a few different Ra

import numpy as np
import h5py
from dedalus import public as de
import matplotlib.pyplot as plt

savefigs=False # set this to False if you don't want to save .png output files with plots

input_direc1='/home/gweddell/analysis/'

inputfile1 = input_direc1+'Ra=1.65e3_time=1500/analysis_s1.h5'

inputfile2 = input_direc1+'Ra=2e3_time=1500/analysis_s1.h5'

inputfile3 = input_direc1+'Ra=7e3_time=200/analysis_s1.h5'


inputfile4 = input_direc1+'Ra=1e4_time=200/analysis_s1.h5'

inputfile5 = input_direc1+'Ra=5e4_time=200/analysis_s1.h5'

inputfile6 = input_direc1+'Ra=1e5_time=200/analysis_s1.h5'

inputfile7 = input_direc1+'Ra=5e5_time=200/analysis_s1.h5'

inputfile8 = input_direc1+'Ra=1e7_time=200/analysis_s1.h5'

#inputfile9 = input_direc1+'Ra=1e8_time=200/analysis_s1.h5'


infilenames = [inputfile1, inputfile2, inputfile3, inputfile4, inputfile5, inputfile6, inputfile7, inputfile8]  #, inputfile3, inputfile4]
versions = [3,3,3,3,3,3,3,4]#,4] #, 3, 3] #, 3]

list_of_time_arrays = []
list_of_iter_arrays = []
list_of_bx_arrays = []
list_of_Lcond_arrays = []
list_of_Lconv_arrays = []
list_of_z_arrays = []
list_of_Re_arrays = []
list_of_Nu_arrays = []
list_of_visc_dissip_arrays = []
list_of_visc_dissip_vort_arrays = []
list_of_gradusq_arrays = []

list_of_wT_arrays = []
for casenum, inputfile in enumerate(infilenames):
    with h5py.File(inputfile, mode='r') as file:
        allscales = list(file['scales'])
        alltasks = list(file['tasks'])
        print("Scales available in this file are ", allscales)
        print("Tasks available in this file are ", alltasks)       
        times = np.array(file['scales']['sim_time'])
        iterations = np.array(file['scales']['iteration'])
        if (versions[casenum] == 3):
            z = np.array(file['scales']['z_hash_2b3e6c1ad6197e7bbb577c37c9be3babe1727daf'])[:]
        elif (versions[casenum] == 4):
            z = np.array(file['scales']['z_hash_d284f597a983ade9304b26b4bf92bfe0a3d1d846'])[:]
        else:
            z = np.array(file['scales']['z/1.0'])[:]
            #z = np.array(file['scales']['z_hash_d284f597a983ade9304b26b4bf92bfe0a3d1d846'])[:]
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

    list_of_time_arrays.append(times)
    list_of_iter_arrays.append(iterations)
    list_of_bx_arrays.append(bx_all)
    list_of_Lcond_arrays.append(L_cond_all_tot)
    list_of_Lconv_arrays.append(L_conv_all)
    list_of_z_arrays.append(z)
    list_of_Re_arrays.append(Re_all)
    list_of_Nu_arrays.append(Nu_all)
    list_of_visc_dissip_arrays.append(visc_dissip_all)
    list_of_visc_dissip_vort_arrays.append(visc_dissip_vort_all)
    list_of_gradusq_arrays.append(gradusq_all)

    
    list_of_wT_arrays.append(wT_all)
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
casenum = 3

L_conv = list_of_Lconv_arrays[casenum]
L_cond = list_of_Lcond_arrays[casenum]
times = list_of_time_arrays[casenum]
z = list_of_z_arrays[casenum]

averaging_interval = 50 # number of free-fall times over which to average
end_time = times[-1]
start_time = end_time - averaging_interval



L_conv_average = get_timeavg(L_conv, times, start_time, end_time)
L_cond_average = get_timeavg(L_cond, times, start_time, end_time)
L_tot_average = L_conv_average+ L_cond_average



fig, ax=plt.subplots()

ax.plot(L_conv_average, z, label=r'$L_{ conv}$')

ax.plot(L_cond_average, z, label=r'$L_{ cond}$')

ax.plot(L_tot_average, z, label=r'$L_{ tot}$')

ax.set_xlabel('Energy flux')
ax.set_ylabel('z')
ax.legend()

if (savefigs):
    figname='flux_balance_onecase.png'
    plt.savefig(figname)

#compare mean temperature profiles in all cases

fig, ax=plt.subplots()

num_cases = len(list_of_time_arrays)
num_freefalltimes_for_average = 50
add_background = False# do we need to add in background state?  Only do this if solved for perturbation
for casenum in range(num_cases):
    # get bx (horizontally averaged T perturbation) for this case, along with z and t arrays
    bx= list_of_bx_arrays[casenum]
    z = list_of_z_arrays[casenum]
    times = list_of_time_arrays[casenum]
    
    #get time-averaged bx over last N free-fall times of run
    
    averaging_interval = num_freefalltimes_for_average 
    end_time = times[-1]
    start_time = end_time - averaging_interval
    bx_average = get_timeavg(bx, times, start_time, end_time)


    if (add_background): 
        # add in the background temperature profile -- remember we are solving for perturbation
        base_state_Tprofile = 1.0 - z
    
        bx_average_withbasestate = bx_average + base_state_Tprofile
    else:
        bx_average_withbasestate = bx_average
    this_label = 'case '+str(casenum)
    ax.plot(z,bx_average_withbasestate, label=this_label)

ax.legend()
ax.set_xlabel(r'$T(z)$')
ax.set_ylabel(r'$z$')

if (savefigs):
    figname='temperature_profiles_varyingRa.png'
    plt.savefig(figname)

#look at time-traces of a few different cases together:

fig, ax=plt.subplots()



num_cases = len(list_of_time_arrays)

for casenum in range(num_cases):
    Re= list_of_Re_arrays[casenum]
    times = list_of_time_arrays[casenum]
    
    this_label = 'case '+str(casenum)
    ax.plot(times, Re, label=this_label)
    
ax.legend(loc="lower center")

ax.set_xlabel('t (in units of free-fall time)')
ax.set_ylabel(r'$Re$')

ax.set_xlim([0.0, 200])

if (savefigs):
    figname='Re_timetraces.png'
    plt.savefig(figname)

#get time-averaged Re and Nu values
from scipy.stats import linregress
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit

Ravalues = np.array([1.65e3, 2e3, 7e3, 1e4, 5e4, 1e5, 5e5,1e7])
#Ravalues = np.array([1.0e5, 1.0e5])
avgRe_values = np.zeros_like(Ravalues)
avgNu_values = np.zeros_like(Ravalues)

num_cases = len(list_of_time_arrays)
if (num_cases != len(Ravalues)):
    print("Number of cases and length of Ravalues/Re/Nu arrays doesn't match!")
num_freefalltimes_for_average = 50
for casenum in range(num_cases):
    # get bx (horizontally averaged T perturbation) for this case, along with z and t arrays
    Re= list_of_Re_arrays[casenum]
    Nu = list_of_Nu_arrays[casenum]
    times = list_of_time_arrays[casenum]
    
    #get time-averaged Re and Nu over last N free-fall times of run
    
    averaging_interval = num_freefalltimes_for_average 
    end_time = times[-1]
    start_time = end_time - averaging_interval
    Re_average = get_timeavg(Re, times, start_time, end_time)
    Nu_average = get_timeavg(Nu, times, start_time, end_time)



    # add to data arrays:
    avgRe_values[casenum] = Re_average
    avgNu_values[casenum] = Nu_average
    print("average Reynolds number for case with Ra ", Ravalues[casenum], " is ", Re_average)
    print("average Nusselt number for case with Ra ", Ravalues[casenum], " is ", Nu_average)
fig, ax=plt.subplots()
ax.plot(Ravalues, avgNu_values, 'o')
ax.set_xscale('log')
ax.set_ylim(0.0, 10.0)
ax.set_xlabel(r'$Ra$')
ax.set_ylabel(r'$Nu$')

print('Nu =',m_opt,'log(Ra) + ',c_opt)

if (savefigs):
    figname='nusselt_rayleigh_example.png'
    plt.savefig(figname)
