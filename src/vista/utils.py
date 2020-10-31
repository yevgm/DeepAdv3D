import numpy as np
import pyvista as pv
import time
import torch

# from geom.tool.vis import parse_color
# from numeric.np import torch2numpy
from vista.color import parse_color
# ---------------------------------------------------------------------------------------------------------------------#
#                            some functions from numeric.np and util.execution for animation
# ---------------------------------------------------------------------------------------------------------------------#


def busy_wait(dt):
    current_time = time.time()
    while time.time() < current_time + dt:
        pass


def torch2numpy(*args):
    out = []
    for arg in args:
        arg = arg.cpu().detach().numpy() if torch.is_tensor(arg) else arg  # added detach().cpu() from ido's function
        out.append(arg)
    if len(args) == 1:
        return out[0]
    else:
        return tuple(out)


# ---------------------------------------------------------------------------------------------------------------------#
#
# ---------------------------------------------------------------------------------------------------------------------#

# ---------------------------------------------------------------------------------------------------------------------#
#
# ---------------------------------------------------------------------------------------------------------------------#
def concat_cell_qualifier(arr):
    return np.concatenate((np.full((arr.shape[0], 1), arr.shape[1]), arr), 1)


def color_to_pyvista_color_params(color, repeats=1):
    color = torch2numpy(color)

    if isinstance(color, str) or len(color) == 3:
        return {'color': parse_color(color)}

    else:
        color = np.asanyarray(color)
        if repeats > 1:
            color = np.repeat(color, axis=0, repeats=repeats)
        return {'scalars': color, 'rgb': color.squeeze().ndim == 2}


# ---------------------------------------------------------------------------------------------------------------------#
#
# ---------------------------------------------------------------------------------------------------------------------#
def plotter(theme='document',**kwargs):
    """
     Plot Menu Controls:

        q                               Close the rendering window
        v                               Isometric camera view
        w                               Switch all datasets to a wireframe representation
        r                               Reset the camera to view all datasets
        s                               Switch all datasets to a surface representation
        shift+click or middle-click     Pan the rendering scene
        left-click                      Rotate the rendering scene in 3D
        ctrl+click                      Rotate the rendering scene in 2D (view-plane)
        mouse-wheel or right-click      Continuously zoom the rendering scene
        shift+s                         Save a screenshot (only on BackgroundPlotter)
        shift+c                         Enable interactive cell selection/picking
        up/down                         Zoom in and out
        +/-                             Increase/decrease the point size and line widths
    """
    pv.set_plot_theme(theme)
    p = pv.Plotter(**kwargs)
    return p
