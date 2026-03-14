import ipywidgets as widgets
import numpy as np
import matplotlib.pyplot as plt

# Define constants (We use atomic units for everything in this demo)
hbar = 1

# Define the interactive widgets

# Quantum number v can be any integer between 0 and 20
v_options = widgets.IntSlider(
    value=0,
    min=0,
    max=20,
    step=1,
    description=r'$v$:',
    disabled=False,
)

# Max Quantum number vmax to be plotted, can be any integer between 0 and 20
vmax_options = widgets.IntSlider(
    value=4,
    min=0,
    max=20,
    step=1,
    description=r'$v_{max}$:',
    disabled=False,
)

# Angular frequency omega can be any float value between 0.01 and 10 (atomic unit) inputed by the user
omega_options = widgets.BoundedFloatText(
    description=r'$\omega$:',                             
    value=2.0,
    min=0.01,
    max=10.0,
    padding = 8)

# Mass m can be any float value between 0.1 and 10 (atomic unit) inputed by the user
m_options = widgets.BoundedFloatText(
    description=r'$m$:',                             
    value=0.5,
    min=0.01,
    max=10.0,
    padding = 8)

# Function that return hermite polynomial Hv(sqrt(alpha)x) for given m, omega,v
def Hv_func(alpha,v):

    # Set up the Hermite polynomial
    domain = [-1,1]
    window = [-np.sqrt(alpha),np.sqrt(alpha)]
    coeff = [0]*v
    coeff.append(1)
    return np.polynomial.hermite.Hermite(coeff,domain=domain, window=window)

fig,(ax1,ax2,ax3) = plt.subplots(3,sharex=True)
plt.subplots_adjust(hspace=0)

@widgets.interact(v=v_options,m=m_options,omega=omega_options)
def update(v,m,omega):
    ax1.cla()
    ax2.cla()
    ax3.cla()

    # Set the range of x values we want to plot
    npoints=2000
    xlower = -10
    xupper = 10

    # obtain the Hermite polynomial fuction
    alpha= m*omega/hbar    
    Hv = Hv_func(alpha,v)

    print("Hv")
    print(Hv)
    # obtain the Hermite polynomial curve data for the range of x values
    x,y=Hv.linspace(npoints,[xlower,xupper])

    print(x)
    print(y)

    # Add a title
    plt.sca(ax1)

    plt.title(r'Harmonic Oscillator'+ '\n'
              r'$m$= '+str(m) + ', $\omega$=' + str(omega) +
              r', $\alpha$=' + str(alpha)+ r', $v$=' + str(v))


    #############################################
    # First work on the Hermite polynomial part #
    #############################################
    # Add y Label
    plt.ylabel(r'$H_v(\sqrt{\alpha}x)$')

    # Plot
    plt.plot(x,y,c='b')
    ymax = 2**(v/2.0)*np.sqrt(float(np.math.factorial(v)))/((alpha/np.pi)**0.25)
    ymin = -ymax
    plt.ylim(ymin,ymax)

    # Set grid
    plt.grid(True)


    #####################
    # Then work on Psi  #
    #####################
    y = 1/(2**(v/2.0)*np.sqrt(float(np.math.factorial(v))))*(alpha/np.pi)**0.25*y*np.exp(-1/2.0*alpha*x**2)

    plt.sca(ax2)

    # Add y Label
    plt.ylabel(r'$\psi(x)$')
    # Plot
    plt.plot(x,y,c='b')   

    # Set grid
    plt.grid(True)

    # Calculate energy of the state with quantum number v
    E = hbar*omega*(v+ 1/2.0)
    # Calculate the positive turning point xt_plus based on E
    xt_plus = np.sqrt(2.0/(m*omega**2)*E)
    xt_minus = -xt_plus  

    # Plot the Classical turning points
    plt.plot([xt_minus, xt_minus],[-1,1],c='k')
    plt.plot([xt_plus,xt_plus],[-1,1],c='k')
    #########################
    # Then  work on Psi**2  #
    #########################

    plt.sca(ax3)
    # Add X and y Label
    plt.xlabel(r'$x$ (bohr)')
    plt.ylabel(r'$|\psi(x)|^2$')

    # Plot
    plt.plot(x,y**2,c='r')
    plt.fill_between(x,y**2, color='r',alpha=0.5)
    plt.ylim(0,1.1)

    # Set grid
    plt.grid(True)

    # Plot the Classical turning points
    plt.plot([xt_minus, xt_minus],[-1,1],c='k')
    plt.plot([xt_plus,xt_plus],[-1,1],c='k')
    plt.text(xt_plus,0.5,'classical turning points')

    plt.show()