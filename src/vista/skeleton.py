import numpy as np
import pyvista as pv
from sklearn.neighbors import NearestNeighbors

# from geom.tool.vis import add_spheres


def ordered_point_connection(v):
    poly = pv.PolyData()
    poly.points = v
    edges = np.full((len(v) - 1, 3), 2)  # Set first column to 2 - For 2 Cells
    edges[:, 1] = np.arange(0, len(v) - 1)  # Set 2nd column to 0->n-1
    edges[:, 2] = np.arange(1, len(v))  # Set 3rd column to 1->n
    poly.lines = edges
    return poly


def nearest_neighbor_point_connection(v, k=2):
    o = NearestNeighbors(n_neighbors=k).fit(v)  # TODO - add in different metrics?
    targets = o.kneighbors(v, n_neighbors=k + 1, return_distance=False)[:, 1:].flatten()  # Remove self match
    sources = np.tile(np.arange(len(v)), (k, 1)).transpose().flatten()
    edges = np.full((len(sources), 3), 2)
    edges[:, 1] = sources
    edges[:, 2] = targets

    poly = pv.PolyData()
    poly.points, poly.lines = v, edges
    return poly


def plot_skeleton(v, edges, tube_radius=0.1, point_size=1,
                  sphere_color=np.array([0, 1, 0]), tube_color=np.array([0, 80, 250]) / 255, transformations=None):
    p = pv.Plotter()
    p, m = add_skeleton(p=p, v=v, edges=edges, tube_radius=tube_radius, point_size=point_size,
                        sphere_color=sphere_color,
                        tube_color=tube_color, transformations=transformations)
    p.show()
    return p, m


def add_skeleton(p, v, edges: np.ndarray, transformations=None, scale=1, plot_points=True, tube_radius=0.01,
                 point_size=1,
                 sphere_color='w', tube_color=np.array([0, 80, 250]) / 255, **kwargs):
    """
    :param p:
    :param v:
    :param plot_points:
    :param transformations:
    :param scale: determines the size of the coordinate system
    :param edges:
    :param float tube_radius: r <0 renders only spheres. r==0 renders lines, and r>0 renders as tubes of radius r
    :param point_size: The sphere size, by units of VTK
    :param [str,nd.arrray] sphere_color: The color of the spheres - either in string or rgb format
    :param [str,nd.arrray] tube_color: The color of the tubes - either in string or rgb format
    :return: The plotter object
    """
    # Construct a PolyData object with vertices + lines filled
    if edges.shape[1] == 2:  # Doesn't have the 2-cell qualifier
        edges = to_pyvista_edges(edges)

    M = pv.PolyData(v)
    M.lines = edges

    if tube_radius > 0:
        M = M.tube(radius=tube_radius)
    if tube_radius >= 0:
        p.add_mesh(M, smooth_shading=True, color=tube_color, lighting=True)

    if plot_points:
        p, _ = add_spheres(p, v, color=sphere_color, radius=point_size, **kwargs)
    if transformations is not None:
        p = add_joint_coordinate_system(p, transformations, scale=scale)
    return p, M


def add_joint_coordinate_system(p, trans_mats, scale=1):
    for joint_transformation in trans_mats:
        vec_start = joint_transformation[0:3, 3]  # Translation Vector
        x_direction = joint_transformation[0:3, 0:3] @ [1, 0.0, 0.0]
        y_direction = joint_transformation[0:3, 0:3] @ [0.0, 1, 0.0]
        z_direction = joint_transformation[0:3, 0:3] @ [0.0, 0.0, 1]
        x_vec = pv.Arrow(start=vec_start, direction=x_direction, tip_length=0.3, tip_radius=0.05, shaft_radius=0.01,
                         shaft_resolution=1, scale=scale)
        y_vec = pv.Arrow(start=vec_start, direction=y_direction, tip_length=0.3, tip_radius=0.05, shaft_radius=0.01,
                         shaft_resolution=1, scale=scale)
        z_vec = pv.Arrow(start=vec_start, direction=z_direction, tip_length=0.3, tip_radius=0.05, shaft_radius=0.01,
                         shaft_resolution=1, scale=scale)

        p.add_mesh(x_vec, color='red')  # R
        p.add_mesh(y_vec, color='green')  # G
        p.add_mesh(z_vec, color='blue')  # B
    return p


def to_pyvista_edges(edges):
    return np.concatenate((np.full((edges.shape[0], 1), 2), edges), 1)