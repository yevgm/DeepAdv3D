from pathlib import Path

import numpy as np
import torch

# from numeric.np.transforms import axangle2aff
# from geom.tool.io.aggregators import meshes_from_dir, multi_meshes_from_dir
# from geom.tool.vis.vista.basic import add_mesh, plot_mesh_montage
# from geom.tool.vis.vista.utils import plotter
# from util.execution import busy_wait
from vista.geom_vis import add_mesh_animation, plot_mesh_montage
from vista.geom_vis import plotter
from vista.utils import concat_cell_qualifier, color_to_pyvista_color_params, busy_wait, torch2numpy

# ---------------------------------------------------------------------------------------------------------------------#
#
# ---------------------------------------------------------------------------------------------------------------------#


def multianimate_from_path(dp: Path, n: int, verts_only=False, **plt_args):
    animations, dir_names = multi_meshes_from_dir(dp, n=n, shared_faces=True, verts_only=verts_only)
    fs = None if verts_only else [anim[1] for anim in animations]
    vss = animations if verts_only else [anim[0] for anim in animations]
    multianimate(vss=vss, fs=fs, titles=[name.stem for name in dir_names], **plt_args)


def multianimate(vss, fs=None, titles=None, color='lightcoral', **plt_args):
    print('Orient the view, then press "q" to start animation')
    p, ms = plot_mesh_montage(vs=[vs[0] for vs in vss], fs=fs, titles=titles, auto_close=False, colors=color,
                              ret_meshes=True, **plt_args)

    num_frames_per_vs = [len(vs) for vs in vss]
    longest_sequence = max(num_frames_per_vs)
    for i in range(longest_sequence):  # Iterate over all frames
        for mi, m in enumerate(ms):
            if i < num_frames_per_vs[mi]:
                p.update_coordinates(points=vss[mi][i], mesh=m, render=False)
                if i == num_frames_per_vs[mi] - 1:
                    for k, actor in p.renderers[mi]._actors.items():
                        if k.startswith('PolyData'):
                            actor.GetProperty().SetColor([0.8] * 3)  # HACKY!
        for renderer in p.renderers:  # TODO - is there something smarter than this?
            renderer.ResetCameraClippingRange()
        p.update()
    p.show(full_screen=False)  # To allow for screen hanging.


def animate_by_path(dp, gif_name=None, pause=0.05, callback_func=None, **plt_args):
    vs, f = meshes_from_dir(dp=dp, shared_faces=True)
    animate(vs, f, gif_name=gif_name, pause=pause, callback_func=callback_func, **plt_args)


def animate(vs, f=None, gif_name=None, first_frame_index=0, pause=0, callback_func=None, color='w', **plt_args):  # was 0.05
    p = plotter()
    first_example = vs[first_frame_index]
    final_colors = (vs[-1].cpu().detach() - first_example.cpu().detach()).norm(p=2, dim=-1)
    clim = [0, torch.max(final_colors)]
    add_mesh_animation(p, vs[first_frame_index], f, color=color, clim=clim, as_a_single_mesh=True, **plt_args)
    # Plot first frame. Normals are not supported
    print('Orient the view, then press "q" to start animation')
    p.show(auto_close=False, full_screen=True)


    # Open a gif
    if gif_name is not None:
        p.open_gif(gif_name)

    for i, v in enumerate(vs):
        if callback_func is not None:
            v = callback_func(p, v, i, len(vs))
        p.update_coordinates(torch2numpy(v), render=False)  # added torch2numpy on v
        new_color = (v.cpu().detach() - first_example.cpu().detach()).norm(p=2, dim=-1)
        p.update_scalars(new_color, render=False)
        # p.reset_camera_clipping_range()
        if pause > 0:
            busy_wait(pause)  # Sleeps crashes the program

        if gif_name is not None:
            p.write_frame()

        if i == len(vs) - 1:
            for k, actor in p.renderer._actors.items():
                actor.GetProperty().SetColor([0.8] * 3)  # HACKY!

        p.update()
    p.show(full_screen=False)  # To allow for screen hanging.


# ---------------------------------------------------------------------------------------------------------------------#
#                                                    Callbacks
# ---------------------------------------------------------------------------------------------------------------------#

def rotation_callback_closure(n_rotations=2, rotation_direction=(0, 1, 0)):
    assert n_rotations > 0

    def rotation_callback(_, v, frame_index, num_frames):
        if frame_index == 0:
            rotation_callback.center = v.mean(axis=0).tolist()

        R = axangle2aff(angle=frame_index * 2 * np.pi * n_rotations / num_frames, axis=rotation_direction,
                            point=rotation_callback.center)[:3, :3]
        return v @ R

    return rotation_callback


# ---------------------------------------------------------------------------------------------------------------------#
#                                                    Test Suite
# ---------------------------------------------------------------------------------------------------------------------#
def _animate_tester():
    from geom.tool.io import DAEFile
    from cfg import FS
    fp = FS.ANIM_ASSET_ROOT / 'Capoeira.dae'
    f = DAEFile(fp)
    f.animate_mesh(max_frames=None, callback_func=rotation_callback_closure())


def _multianimate_tester():
    from cfg import FS
    fp = FS.DATA_ROOT / 'synthetic' / 'DFaust' / 'fulL' / '50002'
    multianimate_from_path(fp, n=16, verts_only=False)


if __name__ == '__main__':
    _animate_tester()
