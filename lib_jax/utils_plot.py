import numpy as np
import matplotlib.pyplot as plt

from matplotlib.collections import LineCollection
from matplotlib.colors import LinearSegmentedColormap


cdict1 = {'red':   ((0.0, 0.0, 0.0),
                    (0.5, 0.0, 0.1),
                    (1.0, 1.0, 1.0)),

         'green': ((0.0, 0.0, 0.0),
                   (1.0, 0.0, 0.0)),

         'blue':  ((0.0, 0.0, 1.0),
                   (0.5, 0.1, 0.0),
                   (1.0, 0.0, 0.0))
         }

blue_red1 = LinearSegmentedColormap('BlueRed1', cdict1, N=256)


def plot_trajectory(particles, colorbar=False, ax=None, fig=None, log=False, ts=None):
    """
        Plot the trajectories of different particles
        
        Inputs:
        - particles: np.array of size (n_time_step, n_particles, d)
    """
    x0 = particles[0]
    
    if ax is None:
        fig, ax = plt.subplots(1,1, figsize=(10,10))


    # ts = np.arange(len(particles))

    if ts is None:
        ts = np.arange(x0.shape[0])
        
    if log:
        ts[0] = 1
        ts = np.log(ts)
    
    for k in range(x0.shape[0]):
        segments = [np.column_stack([particles[i:i+2,k,0], particles[i:i+2,k,1]]) for i in range(len(particles))]   
            
        lc = LineCollection(segments, cmap=blue_red1, array=ts, linewidths=(0.75))
        line = ax.add_collection(lc)
        

    if colorbar:
        cb = fig.colorbar(line, ax=ax)
        cb.ax.set_title("t", fontsize=13)

    ax.scatter(x0[:,0], x0[:,1], label="Initial particles", c="blue")
    ax.scatter(particles[-1][:,0], particles[-1][:,1], label="Final particles", c="red")

    return line
