
# # Homework 7 Solutions
# 
# Rixin Li, H. Foote & G . Besla



# import necessary modules
# numpy provides powerful multi-dimensional arrays to hold and manipulate data
import numpy as np
# matplotlib provides powerful functions for plotting figures
import matplotlib.pyplot as plt
# astropy provides unit system and constants for astronomical calculations
import astropy.units as u
import astropy.constants as const
# import Latex module so we can display the results with symbols
from IPython.display import Latex
get_ipython().run_line_magic('matplotlib', 'inline')

# import the CenterOfMass to determine the COM pos/vel of M33
from CenterOfMass2 import CenterOfMass
# import the GalaxyMass to determine the mass of M31 for each component
from GalaxyMass import ComponentMass


# # M33AnalyticOrbit



class M33AnalyticOrbit:
    '''Class that integrates the orbit of M33 based on the 
    analytic form of M31's potential'''
    
    
    def __init__(self, filename):
        '''
        This class uses leapfrog integration to calculate the orbit of M33 around M31,
        using the analytic potentials for each of M31's components.

        PARAMETERS
        ----------
        filename : `str`
            Name of the file in which to store the orbit
        '''    
        
        # get the gravitational constant (the value is 4.498502151575286e-06)
        self.G = const.G.to(u.kpc**3/u.Msun/u.Gyr**2).value
        
        # store the output file name
        self.filename = filename
        
        # get the COM position and velocity of each galaxy using disk particles
        
        # COM object 
        com_M31 = CenterOfMass('M31_000.txt', 2) # COM object for M31
        com_M33 = CenterOfMass('M33_000.txt', 2) # COM object for M33
        # COM Pos and Velocity for M31
        com_p_M31 = com_M31.COM_P(volDec=2, delta=0.1)
        com_v_M31 = com_M31.COM_V(com_p_M31[0], com_p_M31[1], com_p_M31[2])
        # COM Pos and Velocity for for M33
        com_p_M33 = com_M33.COM_P(volDec=4, delta=0.1)
        com_v_M33 = com_M33.COM_V(com_p_M33[0], com_p_M33[1], com_p_M33[2])

        # set up initial conditions (M33 pos and velocity relative to M31)
        self.r0 = (com_p_M33 - com_p_M31).value # separation vector in kpc
        self.v0 = (com_v_M33 - com_v_M31).value # relative velocity in km/s
             
            
            
        # set up M31 potential parameters
        
        # M31 disk
        self.rdisk = 5.0 # disk scale length in kpc
        self.Mdisk = ComponentMass("M31_000.txt", 2)*1e12 # disk mass in Msun
        
        # M31 bulge
        self.rbulge = 1.0 # bulge scale length in kpc
        self.Mbulge = ComponentMass("M31_000.txt", 3)*1e12 # bulge mass in Msun
        
        # M31 Halo
        self.rhalo = 61.58 # Halo Scale Length: use the Hernquist scale length, 
                    # computed in HW5
        self.Mhalo = ComponentMass("M31_000.txt", 1)*1e12  # halo mass in Msun
     
    
    
    def HernquistAccel(self, M, r_a, r):
        '''
        This method computes the 3-D acceleration due to a Hernquist potential.

        PARAMETERS
        ----------
        M : `float`
            Total mass responsible for the potential in Msun
        r_a : `float`
            Scale length in kpc
        r : `np.ndarray`
            Position vector to find the acceleration at

        RETURNS
        -------
        a : `np.ndarray`
            Acceleration vector in kpc/Gyr^2
        '''

        # follow the formula in the HW instructions
        r_mag = np.sqrt(np.sum(r**2))
        
        # Acceleration for a Hernquist Potential
        # Terms that are constant per component
        M_const = self.G * M / (r_mag * (r_a + r_mag)**2) 
        
        a = -M_const*r # acceleration per component
        
        return a
    
    
    
    def MiyamotoNagaiAccel(self, M, rd, r):
        '''
        This method computes the 3-D acceleration due to a Miyamoto-Nagai disk.

        PARAMETERS
        ----------
        M : `float`
            Disk mass responsible for the potential in Msun
        r_d : `float`
            Disk scale length in kpc
        r : `np.ndarray`
            Position vector to find the acceleration at

        RETURNS
        -------
        a : `np.ndarray`
            Acceleration vector in kpc/Gyr^2
        '''
        # follow the formula for a Miyamoto-Nagai Disk from the HW instructions
       
        # Intermediate Terms
        R2 = np.sum(r[:2]**2)
        zd = rd / 5.0
        B = rd + np.sqrt(r[2]**2 + zd**2)
        
        # constant term in all three components
        M_const = self.G * M / (R2 + B**2)**1.5
        
        # vector of additional terms
        # the np.array allows for a different value for the z component of the acceleration
        vec = np.array([1., 1., B/np.sqrt(r[2]**2 + zd**2)])
        
        # Acceleration
        a = -M_const*vec*r
       
        return a
     
    
    def M31Accel(self, r):
        '''
        This method computes the 3-D acceleration due to M31.

        PARAMETERS
        ----------
        r : `np.ndarray`
            Position vector to find the acceleration at

        RETURNS
        -------
        a : `np.ndarray`
            Acceleration vector in kpc/Gyr^2
        '''
        
        # Compute the acceleration from each galaxy component
        ahalo = self.HernquistAccel(self.Mhalo, self.rhalo, r) 
        abulge = self.HernquistAccel(self.Mbulge, self.rbulge, r) 
        adisk = self.MiyamotoNagaiAccel(self.Mdisk, self.rdisk, r)
    
        a = ahalo+abulge+adisk
        
        return a # return the total acceleration
    
    
    def LeapFrog(self, dt, r, v):
        '''This method is a leapfrog integrator for evolving the position
        and velocity of M33 in time. One call produces one timestep.

        PARAMETERS
        ----------
        dt : `float`
            timestep in Gyr
        r : `np.ndarray`
            position vector at the start of the timestep in kpc
        v : `np.ndarray`
            velocity vector at the start of the timestep in km/s

        RETURNS
        -------
        rnew : `np.ndarray`
            position vector at the end of the timestep in kpc
        vnew : `np.ndarray`
            velocity vector at the end of the timestep in km/s
        '''
        
        # predict the position at the next half timestep
        r_half = r + v * (dt / 2.0)
        
        # compute the velocity at the next timestep
        vnew = v + self.M31Accel(r_half) * dt
        
        # compute the position at the next timestep
        rnew = r + (vnew+v) * dt / 2.0
        
        return rnew, vnew
    
    
    
    def OrbitIntegration(self, t0, dt, tmax):
        '''This method integrates the orbit of M33 and stores the position
            and velocity of M33 (in the COM frame of M31) at each timestep in a file. 

        PARAMETERS
        ----------
        t0 : `float`
            Start time in Gyr
        dt : `float`
            Timestep in Gyr
        tmax : `float`
            End time in Gyr
        ''' 
        
        # find out how many timesteps we have
        N_steps = int((tmax - t0)/dt)+1 # add 1 to account for the integer rounding

        
        # Initializing the time to the input starting time
        t = t0
        
        # Initializing an empty array of size: rows int(tmax/dt)+2  , columns 7
        orbit = np.zeros((int(tmax/dt) + 2, 7))
        
        # Initialize the first row of the orbit
        orbit[0] = t0, *tuple(self.r0), *tuple(self.v0)
        # This above is equivalent to 
        # orbit[0] = t0, self.r0[0], self.r0[1], self.r0[2], self.v0[0], self.v0[1], self.v0[2]
        
        
        # Initializing a counter for the orbit.  
        i = 1 # since we already set the 0th values, we start the counter at 1
        
        # I added this (differs from template)
        r = self.r0 # initial position
        v = self.v0 # initial velocity
        
        
        # Starting the integration (advancing in time steps and computing LeapFrog at each step)
        while (t < tmax): # as long as t has not exceeded the maximal time 
            #print('Integrating Orbit. Running Counter:', i)
            
            # Advancing the time by one timestep, dt
            t += dt
           
            # Storing the new time in the first column of the ith row
            #orbit[i][0] = t
            
            # Advancing the position and velocity using the LeapFrog scheme
            # remember that LeapFrog returns a position vector and a velocity vector  
            # as an example, if a function returns three vectors you would call the function and store 
            # the variable like: a,b,c = function(input)
            
            # This is the position vector that takes the last row of the orbit file. 
            #r = np.array([orbit[i-1][1], orbit[i-1][2], orbit[i-1][3]])
            
            # This is the velocity vector that takes the last row of the orbit file. 
            #v = np.array([orbit[i-1][4], orbit[i-1][5], orbit[i-1][6]])
            
            # This stores the newly integrated position and velocity of the orbit. 
            r, v  = self.LeapFrog(dt, r, v)
    
            # Storing the new position vector into the columns with indexes 1,2,3 of the ith 
            # row of orbit
            # TIP:  if you want columns 5-7 of the Nth row of an array called A, you would write : 
            # A[n, 5:8] 
            # where the syntax is row n, start at column 5 and end BEFORE column 8
              # store system state
            orbit[i] = t, *tuple(r), *tuple(v)
            
            # print progress and increment step counter
            print(f'Finished step {i} of {N_steps}') 
            
            # Storing the new position vector into the columns with indexes 1,2,3 of the ith row of orbit
            #orbit[i][1:4] = r
            #orbit[i][4:7] = v            
           
            # Update counter i , where i is keeping track of the number of rows (i.e. the number of time steps)
            i += 1 

        
        # write the data to a file
        np.savetxt(self.filename, orbit, fmt = "%11.3f"*7, comments='#', 
                   header="{:>10s}{:>11s}{:>11s}{:>11s}{:>11s}{:>11s}{:>11s}"\
                   .format('t', 'x', 'y', 'z', 'vx', 'vy', 'vz'))
        
        # There is no return function

        




# ------------------ FUNCTIONS -------------------# 

# Set the format for reading in the orbit data files   
orbit_type = np.dtype([('t', 'f8'), ('r', 'f8', 3), ('v', 'f8', 3)])


def relative_mag(orbit1, orbit2): 
    """ Function to compute the relative separation and velocity using input orbit files for two galaxies, 
    where the orbit files have `orbit_type` formatting: np.dtype([('t', 'f8'), ('r', 'f8', 3), ('v', 'f8', 3)])

    PARAMETERS
    ----------
    orbit1 : `np.ndarray' 
        first vector, assumed 'orbit_type' formatting
    orbit2 : 'np.ndarray'
        second vector, assumed 'orbit_type' formatting
    RETURNS
    -------
    rel_pos : `float or np.ndarray`
            |pos1 - pos2|
    rel_vel :`float or np.ndarray`
            |vel1 - vel2|
    """
    
    rel_pos = np.sqrt(np.sum((orbit1['r'] - orbit2['r'])**2, axis=1))
    rel_vel = np.sqrt(np.sum((orbit1['v'] - orbit2['v'])**2, axis=1))
    
    return rel_pos,rel_vel
                  
                  
def vector_mag(orbit):
    """ Function to compute the magnitude of the separation and velocity using an input orbit file, 
    where the orbit files have `orbit_type` formatting: np.dtype([('t', 'f8'), ('r', 'f8', 3), ('v', 'f8', 3)])

    PARAMETERS
    ----------
    orbit : `np.ndarray' 
        vector, assumed 'orbit_type' formatting
        
    RETURNS
    --------
    pos : `float or np.ndarray`
         magnitude of the position
    vel : `float or np.ndarray`
         magnitude of the position  
    
    """              
    pos = np.sqrt(np.sum(orbit['r']**2, axis=1)) 
    vel = np.sqrt(np.sum(orbit['v']**2, axis=1))
                      
    return pos, vel




# ------------------- MAIN ---------------------- #

if __name__ == '__main__':

    # make instance of the orbit integrator class
    M33 = M33AnalyticOrbit("M33AnalyticOrbit.txt")

    # Compute the orbit
    M33.OrbitIntegration(0, 0.1, 10.)





    # Read in Orbit of M33 relative to M31 that we computed using LeapFrog
    M33Orbit = np.loadtxt("M33AnalyticOrbit.txt", dtype=orbit_type)





    #Determine the magnitude of the position and velocities of 
    # the galaxies at each point in the orbit
    M31_M33_R, M31_M33_V = vector_mag(M33Orbit)




    # Homework 5 Solutions 

    # Read in simulation Orbit from Homework 6
    M33SimOrbit = np.genfromtxt('Orbit_M33.txt', dtype = orbit_type)
    M31SimOrbit = np.genfromtxt('Orbit_M31.txt', dtype = orbit_type)

    #Determine the magnitude of the position and velocities of 
    # the galaxies at each point in the orbit
    M31_M33_SimR, M31_M33_SimV = relative_mag(M31SimOrbit, M33SimOrbit)




    # Plot the orbital separations of the galaxies 
    #################################

    fig, ax= plt.subplots(figsize=(12, 10))

    # Plot the analytical separation of M31 and M33
    ax.plot(M33Orbit['t'], M31_M33_R, 'b', lw=5, label='M31-M33 Analytic')

    # Plot the simulated separation of M31 and M33
    ax.plot(M33SimOrbit['t'], M31_M33_SimR, 'r', lw=5, label='M31-M33 Simulation')

    # Add axis labels
    ax.set_xlabel('Time (Gyr)', fontsize=22)
    ax.set_ylabel('Separation (kpc)', fontsize=22)
    ax.set_title("Separations vs. Time", fontsize=22)

    #adjust tick label font size
    ax.xaxis.set_tick_params(labelsize=22)
    ax.yaxis.set_tick_params(labelsize=22)

    # add a legend with some customizations.
    legend = ax.legend(loc='upper left',fontsize=20)

    # tight layout
    fig.tight_layout()

    # Save to a file
    fig.savefig('orbit_M33_R.png')


    # In[25]:


    # Plot the orbital velocities of the galaxies 
    #################################

    fig, ax= plt.subplots(figsize=(12, 10))

    # Plot the analytical velocities of M31 and M33
    ax.plot(M33Orbit['t'], M31_M33_V, 'b', lw=5, label='M31-M33 Analytic')

    # Plot the simulated velocities of M31 and M33
    ax.plot(M33SimOrbit['t'], M31_M33_SimV, 'r', lw=5, label='M31-M33 Simulation')

    # Add axis labels
    ax.set_xlabel('Time (Gyr)', fontsize=22)
    ax.set_ylabel('Velocity (km/s)', fontsize=22)
    ax.set_title("Velocities vs. Time", fontsize=22)

    #adjust tick label font size
    ax.xaxis.set_tick_params(labelsize=22)
    ax.yaxis.set_tick_params(labelsize=22)

    # add a legend with some customizations.
    legend = ax.legend(loc='upper left',fontsize=20)

    # tight layout
    fig.tight_layout()

    # Save to a file
    fig.savefig('orbit_M33_V.png')


    # 
    # Q2:  Clearly the plots don't agree after the first pericenter.  
    # 
    # 
    # Q3:  Dynamical Friction !!  The orbit is not decaying - it is not losing energy. Here, dynamical friction refers to the response of the MW's dark matter halo to the passage of M33.  The response can be characterized as a wake, which pulls back gravitationally on M33. This can be treated like a friction term (negative acceleration), causing the orbit of M33 to lose energy. Dynamical friction matters the most in the denser regions of the halo (i.e. at the closest approach of M33), which is why you see the biggest deviations after pericenter. We will go over this in class. 
    # 
    # 
    # Q4:  You could add another set of acceleration terms in LeapFrog  BUT once you include the MW, you have to account for its gravitational influence on M31 and vice versa. In other words you have to compute the orbital evolution of the MW and M31 as well ! 
