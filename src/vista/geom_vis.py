import math
import numpy as np
import pyvista as pv
import torch
from pyvista.plotting.theme import parse_color

# from geom.np.mesh.surface import face_barycenters
# from geom.tool.synthesis import uniform_grid
# from geom.tool.vis import add_skeleton
# from numeric.np import l2_norm_over_last_axis

pv.set_plot_theme('document')  # Change global behaviour - TODO - move this


# ---------------------------------------------------------------------------------------------------------------------#
#                                           General Utility Functions
# ---------------------------------------------------------------------------------------------------------------------#


def plotter(theme='document'):
    pv.set_plot_theme(theme)
    p = pv.Plotter()
    return p


# ---------------------------------------------------------------------------------------------------------------------#
#                                            Visualization Functions
# ---------------------------------------------------------------------------------------------------------------------#
# noinspection PyIncorrectDocstring
def plot_mesh(v, f=None, n=None, strategy='mesh', grid_on=False, clr='lightcoral', normal_clr='lightblue',
              label=None, smooth_shade_on=True, show_edges=False, clr_map='rainbow', normal_scale=1, point_size=None,
              lighting=None, camera_pos=((0, 0, 5.5), (0, 0, 0), (0, 1.5, 0)), opacity=1.0, bar=True, slabel=''
              ,screenshot=False):
    """
    :param v: tensor - A numpy or torch [nv x 3] vertex tensor
    :param f: tensor |  None - (optional) A numpy or torch [nf x 3] vertex tensor OR None
    :param n: tensor |  None - (optional) A numpy or torch [nf x 3] or [nv x3] vertex or face normals. Must input f
    when inputting a face-normal tensor
    :param strategy: One of ['spheres','cloud','mesh']
    :param grid_on: bool - Plots an xyz grid with the mesh. Default is False
    :param clr: str or [R,G,B] float list or tensor - Plots  mesh with color clr. clr = v is cool
    :param normal_clr: str or [R,G,B] float list or tensor - Plots  mesh normals with color normal_clr
    :param label: str - (optional) - When inputted, displays a legend with the title label
    :param smooth_shade_on: bool - Plot a smooth version of the facets - just like 3D-Viewer
    :param show_edges: bool - Show edges in black. Only applicable for strategy == 'mesh'
    For color list, see pyvista.plotting.colors
    * For windows keyboard options, see: https://docs.pyvista.org/plotting/plotting.html
    """
    # White background
    p = plotter()
    p, m = add_mesh(p, v=v, f=f, n=n, grid_on=grid_on, strategy=strategy, clr=clr, normal_clr=normal_clr,
                    label=label, smooth_shade_on=smooth_shade_on, show_edges=show_edges, cmap=clr_map,
                    normal_scale=normal_scale, point_size=point_size, lighting=lighting, camera_pos=camera_pos,
                    opacity=opacity, bar=bar, slabel=slabel)
    p.show()
    return p, m


# noinspection PyIncorrectDocstring
def plot_mesh_montage(vb, fb=None, nb=None, strategy='mesh', labelb=None, grid_on=False, clrb='lightcoral',
                      normal_clr='lightblue', smooth_shade_on=True, show_edges=False, normal_scale=1, auto_close=True,
                      camera_pos=((0, 0, 5.5), (0, 0, 0), (0, 1, 0)), lighting=None,link_plots=True,ext_func=None,
                      opacity=1.0, bar=True, slabelb=None, success=None, classifier_success=None,
                      cmap='rainbow', screenshot=False):
    """
    :param vb: tensor | list - [b x nv x 3] batch of meshes or list of length b with tensors [nvx3]
    :param fb: tensor | list | None - (optional) [b x nf x 3]
    batch of face indices OR a list of length b with tensors [nfx3]
    OR a [nf x 3] in the case of a uniform face array for all meshes
    :param nb: tensor | list | None - (optional) [b x nf|nv x 3]  batch of normals. See above
    :param clrb: list of color options or a single color option
    :param labelb: list of titles for each mesh, or None
    * For other arguments, see plot_mesh
    * For windows keyboard options, see: https://docs.pyvista.org/plotting/plotting.html
    """
    if hasattr(vb, 'shape'):  # Torch Tensor
        n_meshes = tuple(vb.shape)[0]
        vb = vb[:, :, :3]  # Truncate possible normals
    else:
        n_meshes = len(vb)
    assert n_meshes > 0
    n_rows = math.floor(math.sqrt(n_meshes))
    n_cols = math.ceil(n_meshes / n_rows)

    shape = (n_rows, n_cols)
    p = pv.Plotter(shape=shape, off_screen=screenshot)
    r, c = np.unravel_index(range(n_meshes), shape)
    ms = []
    for i in range(n_meshes):
        f = fb if fb is None or (hasattr(fb, 'shape') and fb.ndim == 2) else fb[i]
        if isinstance(clrb, list):
            clr = clrb[i]
        elif isinstance(clrb, np.ndarray):
            if clrb.ndim == 3:
                clr = clrb[:, :, i]
            else:
                clr = clrb
        else:
            clr = clrb
            # Uniform faces support. fb[i] is equiv to fb[i,:,:]
        n = nb if nb is None else nb[i]
        label = labelb if labelb is None else labelb[i]
        opac = opacity[i] if isinstance(opacity, list) else opacity
        slabel = slabelb if slabelb is None else slabelb[i]

        # if it's ground truth don't show colorbar
        if slabel == 'GT':
            cbar = False
        else:
            cbar = bar

        p.subplot(r[i], c[i])
        # if r[i] == c[i]:
        #     p.add_background_image('back.png', as_global=False, auto_resize=True)

        if isinstance(clr, list):
            loop_len = len(clr)
        else:
            loop_len = 1
        for k in np.arange(0,loop_len,1):
            if isinstance(vb[i], list):
                verts = vb[i][k]
                faces = f[k]
                colors = clr[k]
                Opacity = opac[k]
            else:
                verts = vb[i]
                faces = f
                colors = clr
                Opacity = opac

            label1C = classifier_success[i] if classifier_success is not None else None
            label2C = success[i] if success is not None else None
            if label1C is not None:
                labelC = [label1C, label2C]
            else:
                labelC = label2C

            _, m = add_mesh(p, v=verts, f=faces, n=n, strategy=strategy, title=label, grid_on=grid_on,
                            normal_scale=normal_scale, camera_pos=camera_pos, cmap=cmap,
                            clr=colors, normal_clr=normal_clr, smooth_shade_on=smooth_shade_on, show_edges=show_edges,
                            lighting=lighting, opacity=Opacity, bar=cbar, slabel=slabel, label_color=labelC)

        if ext_func is not None:
            ext_func(p, m, i)
        # add_spheres(p,vb[i][1,:][None,:])
        ms.append(m)

    if link_plots:
        p.link_views()

    if screenshot==False:
        p.show(auto_close=auto_close, interactive_update=not auto_close, interactive=auto_close, full_screen=True)

    return p, ms


def plot_projected_vectorfield(v, f, vf, normalize=True, **kwargs):
    p = pv.Plotter()
    add_vectorfield_tangent_projection(p=p, v=v, f=f, vf=vf, normalize=normalize, **kwargs)
    p.show()


# ---------------------------------------------------------------------------------------------------------------------#
#                                                   Additional Functions
# ---------------------------------------------------------------------------------------------------------------------#


def add_vectorfield_tangent_projection(p, v, f, vf, normalize=True, **kwargs):
    if vf.shape[0] == 3 * f.shape[0]:
        vf = np.reshape(vf, (3, f.shape[0])).T
    from geom.np.mesh.descriptor import tangent_projection
    vfp, clr = tangent_projection(v, f, vf, normalize)
    M = add_mesh(p, v=v, f=f, n=vfp, clr=clr, strategy='mesh', **kwargs)
    return p, M


def add_spheres(p, v, color='black', radius=1, resolution=40, **kwargs):
    src = pv.PolyData(v)
    # spherical2cartesian(np.random.random((n, 3)) * scale)
    src["radius"] = radius * np.ones_like(np.asanyarray(radius))
    geom = pv.Sphere(theta_resolution=resolution, phi_resolution=resolution)
    glyphed = src.glyph(scale="radius", geom=geom)
    p.add_mesh(glyphed, color=color, **kwargs)  # TODO - handle add_mesh discrepency
    return p


def add_integer_vertex_labels(p, v, font_size=10):
    M = pv.PolyData(v)
    # TODO - show_points doesn't seem to work
    p.add_point_labels(M, [str(i) for i in range(v.shape[0])], font_size=font_size, show_points=False,
                       shape=None, render_points_as_spheres=False, point_size=-1)
    p.add_floor()
    return p


def add_isocontours(p, v, f, scalar_func, isosurfaces=15, color='black', line_width=3, opacity=0.5):
    pnt_cloud = pv.PolyData(v, np.concatenate((np.full((f.shape[0], 1), 3), f), 1))
    pnt_cloud._add_point_array(scalars=scalar_func, name='Func')
    contours = pnt_cloud.contour(isosurfaces=isosurfaces, scalars='Func')
    m = p.add_mesh(contours, color=color, line_width=line_width, smooth_shading=True, opacity=opacity,
                   render_lines_as_tubes=True)
    return p, m


def add_floor_grid(p, nx=30, ny=30, x_bounds=(-3, 3), y_bounds=(-3, 3), z_val=0, plot_points=False,
                   grid_color='wheat', with_axis=False, **kwargs):
    v, edges = uniform_grid(nx=nx, ny=ny, x_bounds=x_bounds, y_bounds=y_bounds, z_val=z_val, cell_type='edges')

    if with_axis:
        axis = [np.array([[1, 0, 0, x_bounds[0]], [0, 1, 0, y_bounds[0]], [0, 0, 1, z_val + 0.1], [0, 0, 0, 1]])]
    else:
        axis = None
    return add_skeleton(p, v=v, edges=edges, plot_points=plot_points, tube_color=grid_color, transformations=axis,
                        tube_radius=0.02, **kwargs)


def add_vectorfield(p, v, f, vf, clr='lightblue', normal_scale=1, colormap='rainbow'):
    # TODO - Support explicit supply of color per arrow - need to extend clr vector on to arrow mesh(with np.tile)

    # Align flat vector fields:
    if f is not None and vf.shape[0] == 3 * f.shape[0]:
        vf = np.reshape(vf, (3, f.shape[0])).T
    elif vf.shape[0] == 3 * v.shape[0]:
        vf = np.reshape(vf, (3, v.shape[0])).T

    # Prepare the PolyData object:
    if f is not None and vf.shape[0] == f.shape[0]:
        pnt_cloud = pv.PolyData(face_barycenters(v, f))
    else:
        assert vf.shape[0] == v.shape[0]
        pnt_cloud = pv.PolyData(v)

    pnt_cloud['glyph_scale'] = l2_norm_over_last_axis(vf)
    pnt_cloud['vectors'] = vf
    arrows = pnt_cloud.glyph(orient="vectors", scale="glyph_scale", factor=0.05 * normal_scale)
    p.add_mesh(arrows, color=clr, colormap=colormap)
    return p, arrows


def add_mesh(p, v, f=None, n=None, strategy='spheres', grid_on=False, clr='lightcoral',
             normal_clr='lightblue', title=None, smooth_shade_on=True, show_edges=False, cmap='rainbow',
             normal_scale=1, camera_pos=((0, 0, 5.5), (0, 0, 0), (0, 1.5, 0)), lines=None, opacity=1.0,
             point_size=None, lighting=None, eye_dome=False, bar=True, slabel='', label_color=None):
    # TODO - Clean this shit function up
    # Align arrays:
    cpu = torch.device("cpu")
    v = v.to(cpu).clone().detach().numpy() if torch.is_tensor(v) else v
    f = f.to(cpu).clone().detach().numpy() if torch.is_tensor(f) else f
    n = n.to(cpu).clone().detach().numpy() if torch.is_tensor(n) else n
    clr = clr.to(cpu).clone().detach().numpy() if torch.is_tensor(clr) else clr
    normal_clr = normal_clr.to(cpu).clone().detach().numpy() if torch.is_tensor(normal_clr) else normal_clr

    # Align strategy
    style = 'surface'
    if strategy == 'mesh' or strategy == 'wireframe':
        assert f is not None, "Must supply faces for mesh strategy"
        if strategy == 'wireframe':
            style = strategy
    else:
        f = None  # Destroy the face information
    spheres_on = (strategy == 'spheres')

    # Create Data object:
    if f is not None:
        # Adjust f to the needed format
        pnt_cloud = pv.PolyData(v, np.concatenate((np.full((f.shape[0], 1), 3), f), 1))
    else:
        pnt_cloud = pv.PolyData(v)

    if lines is not None:
        pnt_cloud.lines = lines

    # Default size for spheres & pnt clouds
    if point_size is None:
        point_size = 12.0 if spheres_on else 2.0  # TODO - Dynamic computation of this, based on mesh volume

    # Handle difference between color and scalars, to support RGB tensor
    if isinstance(clr, str) or len(clr) == 3:
        scalars = np.tile(parse_color(clr), (v.shape[0], 1))
        clr_str = clr
        rgb = True
    else:
        if isinstance(label_color, list): #& (label_color[0] is not None):
            if label_color[0] == True:
                original_color = 'g'
            else:
                original_color = 'r'
            if label_color[1] == True:
                perturbed_color = 'g'
            else:
                perturbed_color = 'r'
        else:
            if label_color is True:
                clr_str = 'g'
            elif label_color is False:
                clr_str = 'r'
            else:
                clr_str = 'b'
        scalars = clr
        rgb = isinstance(clr, (np.ndarray, np.generic)) and clr.squeeze().ndim == 2  # RGB Vector
    # TODO - use a kwargs approach to solve messiness
    if lighting is None:
        # Default Pyvista Light
        d_light = {'ambient': 0.0, 'diffuse': 1.0, 'specular': 0.0, 'specular_power': 100.0}
    else:
        # Our Default Lighting
        d_light = {'ambient': 0.2, 'diffuse': 0.6, 'specular': 1, 'specular_power': 2}

    # scalar-bar arguments
    sargs = dict(height=0.45, vertical=True, position_x=0.15, position_y=0.15, width=0.05)

    # Add the meshes to the plotter:
    p.add_mesh(pnt_cloud, style=style, smooth_shading=smooth_shade_on, scalars=scalars, cmap=cmap,
               show_edges=show_edges,  # For full mesh visuals - ignored on point cloud plots
               render_points_as_spheres=spheres_on, point_size=point_size,
               rgb=rgb, opacity=opacity, lighting=lighting, scalar_bar_args=sargs,
               stitle=slabel, show_scalar_bar=bar,
               **d_light)  # For sphere visuals - ignored on full mesh

    # ambient/diffuse/specular strength, specular exponent, and specular color
    #     'Shiny',	0.3,	0.6,	0.9,	20,		1.0
    #     'Dull',		0.3,	0.8,	0.0,	10,		1.0
    #     'Metal',	0.3,	0.3,	1.0,	25,		.5
    if camera_pos is not None:
        p.camera_position = camera_pos
    if n is not None:  # Face normals or vertex normals
        add_vectorfield(p, v, f, n, clr=normal_clr, normal_scale=normal_scale)

    # Book-keeping:
    # if label is not None and label:
    #     siz = 0.2
    #     p.add_legend(labels=[(label, clr_str)], size=[siz, siz / 2], bcolor=(1, 1, 1))
    if isinstance(title, list):
        p.add_text(title[0], font_size=9, position='upper_edge', color=original_color)
        p.add_text('\n'+title[1], font_size=9, position='upper_edge', color=perturbed_color)
    else:
        p.add_text(title, font_size=11, position='upper_edge', color=clr_str)
    if grid_on:
        p.show_grid()

    if eye_dome:
        p.enable_eye_dome_lighting()
    return p, pnt_cloud


# ---------------------------------------------------------------------------------------------------------------------#
#                                                    Helper Functions
# ---------------------------------------------------------------------------------------------------------------------#


# ---------------------------------------------------------------------------------------------------------------------#
#                                                    Test Suite
# ---------------------------------------------------------------------------------------------------------------------#

# ---------------------------------------------------------------------------------------------------------------------#
#
# ---------------------------------------------------------------------------------------------------------------------#


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
    shift+s                         Save a screenhsot (only on BackgroundPlotter)
    shift+c                         Enable interactive cell selection/picking
    up/down                         Zoom in and out
    +/-                             Increase/decrease the point size and line widths
"""
# ---------------------------------------------------------------------------------------------------------------------#
#
# ---------------------------------------------------------------------------------------------------------------------#
if __name__ == '__main__':
    pass
