from typing import Union

import numpy as np
from pyvista.plotting.colors import string_to_rgb


# # https://matplotlib.org/3.1.0/tutorials/colors/colormaps.html
# ---------------------------------------------------------------------------------------------------------------------#
#
# ---------------------------------------------------------------------------------------------------------------------#

def color_tile(N, clr='w', inds=None, clr2='r'):
    rgb = np.tile(parse_color(clr), (N, 1))
    if inds is not None:
        rgb[inds, :] = np.tile(parse_color(clr2), (len(inds), 1))
    return rgb


def parse_color(color: Union[str, list, tuple, np.ndarray] = 'cyan', opacity: Union[int, float] = None):
    """Parse color into a vtk friendly rgb list.
    Values returned will be between 0 and 1.
    """
    if isinstance(color, str):
        color = list(string_to_rgb(color))
    elif len(color) == 3 or len(color) == 4:
        color = list(color)  # Handle numpy arrays
    else:
        raise ValueError(f"Invalid color input: ({color} "
                         f"Must be string, rgb list, or hex color string.  "
                         f"Examples: color='white'/ 'w' / [1, 1, 1] / '#FFFFFF'")
    # Handle opacity
    if opacity is not None:
        assert isinstance(opacity, (float, int))
        color.insert(3, opacity)

    if any([c > 1 for c in color]):
        assert sum([c > 255 for c in color]) == 0, f"Invalid color values in range: {color}"
        color = [c / 255 for c in color]
    else:
        color = [float(c) for c in color]
    return color


# ---------------------------------------------------------------------------------------------------------------------#
#
# ---------------------------------------------------------------------------------------------------------------------#

def _parse_color_tester():
    print(parse_color('r'))
    print(parse_color('red'))
    print(parse_color([1, 0, 0]))
    print(parse_color([255, 0, 0]))
    print(parse_color([1, 0, 0, 1]))
    print(parse_color([255, 0, 0, 10]))
    # print(parse_color([256,0,0]))

    print(parse_color(np.asarray([1, 0, 0])))
    print(parse_color(np.asarray([255, 0, 0])))
    print(parse_color(np.asarray([1, 0, 0, 1])))
    print(parse_color(np.asarray([255, 0, 0, 10])))
    # print(parse_color(np.asarray([256,0,0])))

    print(parse_color('r', opacity=1))
    print(parse_color('red', opacity=1))
    print(parse_color([1, 0, 0], opacity=1))
    print(parse_color([255, 0, 0], opacity=1))
    print(parse_color([1, 0, 0, 1], opacity=1))
    print(parse_color([255, 0, 0, 10], opacity=1))
    # print(parse_color([256,0,0]))

    print(parse_color(np.asarray([1, 0, 0]), opacity=1))
    print(parse_color(np.asarray([255, 0, 0]), opacity=1))
    print(parse_color(np.asarray([1, 0, 0, 1]), opacity=1))
    print(parse_color(np.asarray([255, 0, 0, 10]), opacity=1))


if __name__ == '__main__':
    _parse_color_tester()
