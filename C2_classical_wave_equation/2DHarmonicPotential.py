import ipywidgets as widgets
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits import mplot3d

def V(x, y, k11, k22, k12):
    return 0.5*k11*x ** 2 + 0.5*k22*y ** 2 + k12*x*y

# Define the interactive widgets

k11_options = widgets.FloatSlider(
    value=1.0,
    min=0.1,
    max=2.0,
    step=0.1,
    description=r'$k_{11}$:',
    disabled=False,
)

k22_options = widgets.FloatSlider(
    value=1.0,
    min=0.1,
    max=2.0,
    step=0.1,
    description=r'$k_{22}$:',
    disabled=False,
)

k12_options = widgets.FloatSlider(
    value=0,
    min=-1.0,
    max=1.0,
    step=0.1,
    description=r'$k_{12}$:',
    disabled=False,
)

y_options = widgets.FloatSlider(
    value=0,
    min=-6,
    max=6,
    step=0.5,
    description=r'$Y$:',
    disabled=False,
)

x_options = widgets.FloatSlider(
    value=0,
    min=-6,
    max=6,
    step=0.5,
    description=r'$X$:',
    disabled=False,
)


theta_options = widgets.FloatSlider(
    value=0,
    min=0,
    max=180,
    step=10,
    description=r'$\theta$:',
    disabled=False,
)

fig = plt.figure()

@widgets.interact(k11=k11_options,k22=k22_options, k12=k12_options)
def update(k11,k22,k12):
    fig.clf()

    x = np.linspace(-6, 6, 30)
    y = np.linspace(-6, 6, 30)

    X, Y = np.meshgrid(x, y)
    Z = V(X, Y, k11, k22, k12)

    ax = plt.axes(projection='3d')
    ax.contour3D(X, Y, Z, 50, cmap='viridis')
    ax.set_xlabel('x')
    ax.set_ylabel('y')
    ax.set_zlabel('V(x,y)');
    plt.show()


