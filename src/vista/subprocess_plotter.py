import pyvista as pv
from vista.geom_vis import vertex_mask_indicator
from vista.geom_vis import mesh_append
from multiprocessing import Process, Manager
from copy import deepcopy
from abc import ABC
import numpy as np

# ----------------------------------------------------------------------------------------------------------------------#
#                                               Parallel Plot suite
# ----------------------------------------------------------------------------------------------------------------------#

class ParallelPlotterBase(Process, ABC):

    def __init__(self, run_config):
        super().__init__()
        self.run_config = run_config
        self.sd = Manager().dict()  # shared data between processes (vertices, faces .. )
        # self.lock = Manager().Lock()
        # self.lock.acquire()
        self.sd['epoch'] = -1
        self.sd['poison'] = False

        self.last_plotted_epoch = -1
        self.train_d, self.val_d, self.data_cache, self.plt_title = None, None, None, None

        self.kwargs = {'smooth_shade_on': run_config['VIS_SMOOTH_SHADING'], 'show_edges': run_config['VIS_SHOW_EDGES'],
                       'strategy': run_config['VIS_STRATEGY'], 'cmap': run_config['VIS_CMAP'],
                       'grid_on': run_config['VIS_SHOW_GRID']}

    def run(self):
        # Init on consumer side:
        pv.set_plot_theme("document")
        while 1:
            try:
                if self.last_plotted_epoch != -1 and self.sd['poison']:  # Plotted at least one time + Poison
                    print(f'Pipe poison detected. Displaying one last time, and then exiting plotting supervisor')
                    self.try_update_data(final=True)
                    self.plot_data()
                    break
            except (BrokenPipeError, EOFError):  # Missing parent
                print(f'Producer missing. Exiting plotting supervisor')
                break
            # self.lock.acquire()
            self.try_update_data()
            self.plot_data()



    # Meant to be called by the consumer ( the visualizer )
    def try_update_data(self, final=False):
        current_epoch = self.sd['epoch']
        if current_epoch != self.last_plotted_epoch:
            self.last_plotted_epoch = current_epoch
            self.train_d, self.val_d = deepcopy(self.sd['data'])
        if final:
            self.plt_title = f'Final visualization before closing for Epoch {self.last_plotted_epoch}'
        else:
            self.plt_title = f'Visualization for Epoch {self.last_plotted_epoch}'

            # Update version with one single read
        # Slight problem of atomicity here - with respect to current_epoch. Data may have changed in the meantime -
        # but it is still new data, so no real problem. May be resolved with lock = manager.Lock() before every update->
        # lock.acquire() / lock.release(), but this will be problematic for the main process - which we want to optimize

    # Meant to be called by the consumer ( the visualizer )
    def plot_data(self):
        raise NotImplementedError

    # Meant to be called by the producer ( the train loop )
    def push(self, new_epoch, new_data):
        old_epoch = self.sd['epoch']
        assert new_epoch != old_epoch

        # Update shared data (possibly before process starts)
        self.sd['data'] = new_data
        self.sd['epoch'] = new_epoch

        if old_epoch == -1:  # First push
            self.start()

        # self.lock.release()

    def cache(self, data):
        self.data_cache = data

    def uncache(self):
        cache = self.data_cache
        self.data_cache = None
        return cache
        # Meant to be called by the producer ( the train loop )

    def finalize(self):
        self.sd['poison'] = True
        print('Workload completed - Please exit plotter to complete execution')
        self.join()


# ----------------------------------------------------------------------------------------------------------------------#
#                                               Parallel Plot suite
# ----------------------------------------------------------------------------------------------------------------------#

class AdversarialPlotter(ParallelPlotterBase):
    def __init__(self, run_config):
        super().__init__(run_config)

    def prepare_plotter_dict(self, orig_vertices, adexs, faces):

        max_b_idx = self.VIS_N_MESH_SETS
        dict = {'orig_vertices': orig_vertices.detach().cpu().numpy()[:max_b_idx, :, :],
                'adexs': adexs.detach().cpu().numpy()[:max_b_idx, :, :],
                'faces': faces.detach().cpu().numpy()[:max_b_idx]}
        return dict

    def plot_data(self):
        run_config = self.run_config
        p = pv.Plotter(shape=(2 * run_config['VIS_N_MESH_SETS'], 2), title=self.plt_title)
        for di, (d, set_name) in enumerate(zip([self.train_d, self.val_d], ['Train', 'Vald'])):
            for i in range(run_config['VIS_N_MESH_SETS']):
                subplt_row_id = i + di * run_config['VIS_N_MESH_SETS']
                orig_vertices = d['orig_vertices'][i].squeeze().transpose()
                adex = d['adexs'][i].squeeze().transpose()
                faces = d['faces'][i].squeeze()
                diff = np.linalg.norm((orig_vertices - adex), ord=2, axis=1)
                zeros = np.zeros(diff.shape)

                p.subplot(subplt_row_id, 0)  # original example
                mesh_append(p, run_config=run_config, v=orig_vertices, f=faces,
                            clr=zeros, label=f'{set_name} original {i}', **self.kwargs)
                p.subplot(subplt_row_id, 1)  # adversarial example
                mesh_append(p, run_config=run_config, v=adex, f=faces,
                            clr=diff, label=f'{set_name} Adv example {i}', **self.kwargs)
                # if self.last_plotted_epoch > 0:
                #     p.update(force_redraw=True)

        # if self.last_plotted_epoch == 0:
        # p.link_views()
        p.show(auto_close=False)
        # elif self.last_plotted_epoch > 0:
        #     p.update(force_redraw=True)
        # if self.last_plotted_epoch > 0:
        #     p.update(force_redraw=True)
