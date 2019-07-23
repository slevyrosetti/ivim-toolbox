#!/usr/bin/env python
# -*- coding: utf-8 -*-
#########################################################################################
#
# Display 3D image in axial view and enable user to inspect fitting quality by clicking on
# any voxel and directly display the corresponding fit plot
#
# ---------------------------------------------------------------------------------------
# This script was largely inspired from the script "sct_viewer.py" of the Spinal Cord Toolbox
# project (https://github.com/neuropoly/spinalcordtoolbox)
# De Leener B, Levy S, Dupont SM, Fonov VS, Stikov N, Louis Collins D, Callot V, Cohen-Adad J.,
# SCT: Spinal Cord Toolbox, an open-source software for processing spinal cord MRI data. Neuroimage 2017.
# https://www.ncbi.nlm.nih.gov/pubmed/27720818
#
# I would like to truly thank the Spinal Cord Toolbox team, and in particular Benjamin De Leener and
# Julien Cohen-Adad, for developing this script!
#
#########################################################################################

import sys, os
from time import time
from copy import copy
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
from matplotlib.lines import Line2D
from matplotlib import cm
import nibabel as nib
import argparse


class SinglePlot:
    """
        This class manages mouse events on one image.
    """
    def __init__(self, ax, images, viewer, view=2, display_cross='hv', im_params=None):
        self.axes = ax
        self.images = images  # this is a list of images
        self.viewer = viewer
        self.view = view
        self.display_cross = display_cross
        self.image_dim = self.images[0].shape
        self.figs = []
        self.cross_to_display = None
        self.aspect_ratio = None
        self.zoom_factor = 1.0

        for i, image in enumerate(images):
            data_to_display = self.set_data_to_display(image)
            (my_cmap, my_interpolation, my_alpha, clim) = self.set_image_parameters(im_params,i,cm)
            my_cmap.set_under('b', alpha=0)
            self.figs.append(self.axes.imshow(data_to_display, origin="lower", aspect=self.aspect_ratio, alpha=my_alpha, clim=clim))
            self.figs[-1].set_cmap(my_cmap)
            self.figs[-1].set_interpolation(my_interpolation)

        self.axes.set_facecolor('black')
        self.axes.set_xticks([])
        self.axes.set_yticks([])

        self.draw_line(display_cross)


    def draw_line(self,display_cross):
        self.line_horizontal = Line2D(self.cross_to_display[1][1], self.cross_to_display[1][0], color='white')
        self.line_vertical = Line2D(self.cross_to_display[0][1], self.cross_to_display[0][0], color='white')
        if 'h' in display_cross:
            self.axes.add_line(self.line_horizontal)
        if 'v' in display_cross:
            self.axes.add_line(self.line_vertical)

    def set_image_parameters(self,im_params,i,cm):
        if str(i) in im_params.images_parameters:
            return (copy(cm.get_cmap(im_params.images_parameters[str(i)].cmap)),im_params.images_parameters[str(i)].interp,float(im_params.images_parameters[str(i)].alpha), np.array(im_params.images_parameters[str(i)].clim.split(';'), dtype=float))
        else:
            return (cm.get_cmap('gray'), 'nearest', 1.0, (0, 1))

    def set_data_to_display(self,image):
        if self.view == 1:
            self.cross_to_display = [[[self.viewer.current_point.y, self.viewer.current_point.y], [-10000, 10000]],
                                     [[-10000, 10000], [self.viewer.current_point.z, self.viewer.current_point.z]]]
            self.aspect_ratio = self.viewer.aspect_ratio[0]
            return image.data[int(self.image_dim[0] / 2), :, :]
        elif self.view == 2:
            self.cross_to_display = [[[self.viewer.current_point.x, self.viewer.current_point.x], [-10000, 10000]],
                                     [[-10000, 10000], [self.viewer.current_point.z, self.viewer.current_point.z]]]
            self.aspect_ratio = self.viewer.aspect_ratio[1]
            return image.data[:, int(self.image_dim[1] / 2), :]
        elif self.view == 3:
            self.cross_to_display = [[[self.viewer.current_point.x, self.viewer.current_point.x], [-10000, 10000]],
                                     [[-10000, 10000], [self.viewer.current_point.y, self.viewer.current_point.y]]]
            self.aspect_ratio = self.viewer.aspect_ratio[2]
            return image.data[:, :, int(self.image_dim[2] / 2)]
        elif self.view == 4:
            self.cross_to_display = [[[self.viewer.current_point.x, self.viewer.current_point.x], [-10000, 10000]],
                                     [[-10000, 10000], [self.viewer.current_point.y, self.viewer.current_point.y]]]
            self.aspect_ratio = self.viewer.aspect_ratio[2]
            return image.get_data()[:, :, 0]

    def connect(self):
        """
        connect to all the events we need
        :return:
        """
        self.cidpress_click = self.figs[0].figure.canvas.mpl_connect('button_press_event', self.on_press)
        self.cidscroll = self.figs[0].figure.canvas.mpl_connect('scroll_event', self.on_scroll)
        self.cidrelease = self.figs[0].figure.canvas.mpl_connect('button_release_event', self.on_release)
        self.cidmotion = self.figs[0].figure.canvas.mpl_connect('motion_notify_event', self.on_motion)

    def draw(self):
        self.figs[0].figure.canvas.draw()

    def update_slice(self, target, data_update=True):
        """
        This function change the viewer to update the current slice
        :param target: number of the slice to go on
        :param data_update: False if you don't want to update data
        :return:
        """
        if isinstance(target, list):
            target_slice = target[self.view - 1]
            list_remaining_views = list([0, 1, 2])
            list_remaining_views.remove(self.view - 1)
            self.cross_to_display[0][0] = [target[list_remaining_views[0]], target[list_remaining_views[0]]]
            self.cross_to_display[1][1] = [target[list_remaining_views[1]], target[list_remaining_views[1]]]
        else:
            target_slice = target

        # if 0 <= target_slice < self.images[0].data.shape[int(self.view)-1]:
        if 0 <= target_slice < self.image_dim[2]:
            if data_update:
                for i, image in enumerate(self.images):
                    if self.view == 1:
                        self.figs[i].set_data(image.data[target_slice, :, :])
                    elif self.view == 2:
                        self.figs[i].set_data(image.data[:, target_slice, :])
                    elif self.view == 3:
                        self.figs[i].set_data(image.data[:, :, target_slice])
                    elif self.view == 4:
                        self.figs[i].set_data(image.get_data()[:, :, target_slice].T)
            self.set_line_to_display()
        self.figs[0].figure.canvas.draw()

    def set_line_to_display(self):
        if 'v' in self.display_cross:
            self.line_vertical.set_ydata(self.cross_to_display[0][0])
        if 'h' in self.display_cross:
            self.line_horizontal.set_xdata(self.cross_to_display[1][1])

    def on_press(self, event):
        """
        when pressing on the screen, add point into a list, then change current slice
        if finished, close the window and send the result
        :param event:
        :return:
        """
        if event.button == 1 and event.inaxes == self.axes:
            self.viewer.on_press(event, self)

        return

    def change_intensity(self, min_intensity, max_intensity, id_image=0):
        self.figs[id_image].set_clim(min_intensity, max_intensity)
        self.figs[id_image].figure.canvas.draw()

    def on_motion(self, event):
        if event.button == 1 and event.inaxes == self.axes:
            return self.viewer.on_motion(event, self)

        elif event.button == 3 and event.inaxes == self.axes:
            return self.viewer.change_intensity(event, self)

        else:
            return

    def on_release(self, event):
        if event.button == 1:
            return self.viewer.on_release(event, self)

        elif event.button == 3:
            return self.viewer.change_intensity(event, self)

        else:
            return

    def update_xy_lim(self, x_center=None, y_center=None, x_scale_factor=1.0, y_scale_factor=1.0, zoom=True):
        # get the current x and y limits
        cur_xlim = self.axes.get_xlim()
        cur_ylim = self.axes.get_ylim()

        if x_center is None:
            x_center = (cur_xlim[1] - cur_xlim[0]) / 2.0
        if y_center is None:
            y_center = (cur_ylim[1] - cur_ylim[0]) / 2.0

        # Get distance from the cursor to the edge of the figure frame
        x_left = x_center - cur_xlim[0]
        x_right = cur_xlim[1] - x_center
        y_top = y_center - cur_ylim[0]
        y_bottom = cur_ylim[1] - y_center

        if zoom:
            scale_factor = (x_scale_factor + y_scale_factor) / 2.0
            if 0.005 < self.zoom_factor * scale_factor <= 3.0:
                self.zoom_factor *= scale_factor

                self.axes.set_xlim([x_center - x_left * x_scale_factor, x_center + x_right * x_scale_factor])
                self.axes.set_ylim([y_center - y_top * y_scale_factor, y_center + y_bottom * y_scale_factor])
                self.figs[0].figure.canvas.draw()
        else:
            self.axes.set_xlim([x_center - x_left * x_scale_factor, x_center + x_right * x_scale_factor])
            self.axes.set_ylim([y_center - y_top * y_scale_factor, y_center + y_bottom * y_scale_factor])
            self.figs[0].figure.canvas.draw()

    def on_scroll(self, event):
        """
        when scrooling with the wheel, image is zoomed toward position on the screen
        :param event:
        :return:
        """
        if event.inaxes == self.axes:
            base_scale = 0.5
            xdata, ydata = event.xdata, event.ydata

            if event.button == 'up':
                # deal with zoom in
                scale_factor = 1 / base_scale
            elif event.button == 'down':
                # deal with zoom out
                scale_factor = base_scale
            else:
                # deal with something that should never happen
                scale_factor = 1.0
                print(event.button)

            self.update_xy_lim(x_center=xdata, y_center=ydata,
                               x_scale_factor=scale_factor, y_scale_factor=scale_factor,
                               zoom=True)

        return

class Viewer(object):
    def __init__(self, list_input, visualization_parameters=None):
        self.images = self.keep_only_images(list_input)
        self.im_params = visualization_parameters

        """ Initialisation of plot """
        self.fig = plt.figure(os.path.abspath(self.images[0].file_map['image'].filename), figsize=(8, 8))
        self.fig.subplots_adjust(bottom=0.1, left=0.1)
        self.fig.patch.set_facecolor('white')

        """ Pad the image so that it is square in axial view (useful for zooming) """
        self.image_dim = self.images[0].shape
        nx, ny, nz = self.images[0].header.get_data_shape()
        px, py, pz = self.images[0].header.get_zooms()
        self.im_spacing = [px, py, pz]
        self.aspect_ratio = [float(self.im_spacing[1]) / float(self.im_spacing[2]),
                             float(self.im_spacing[0]) / float(self.im_spacing[2]),
                             float(self.im_spacing[0]) / float(self.im_spacing[1])]
        self.offset = [0.0, 0.0, 0.0]
        self.current_point = Coordinate([int(nx / 2), int(ny / 2), int(nz / 2)])

        self.windows = []
        self.press = [0, 0]

        self.mean_intensity = []
        self.std_intensity = []

        self.last_update = time()
        self.update_freq = 1.0/15.0  # 10 Hz

    def keep_only_images(self,list_input):
        # TODO: check same space
        # TODO: check if at least one image
        images=[]
        for im in list_input:
            if isinstance(im, nib.Nifti1Image):
                images.append(im)
            else:
                sys.exit("Error, one of the images is actually not an image...")
        return images

    def compute_offset(self):
        array_dim = [self.image_dim[0]*self.im_spacing[0], self.image_dim[1]*self.im_spacing[1], self.image_dim[2]*self.im_spacing[2]]
        index_max = np.argmax(array_dim)
        max_size = array_dim[index_max]
        self.offset = [int(round((max_size - array_dim[0]) / self.im_spacing[0]) / 2),
                       int(round((max_size - array_dim[1]) / self.im_spacing[1]) / 2),
                       int(round((max_size - array_dim[2]) / self.im_spacing[2]) / 2)]

    def pad_data(self):
        for i_img in range(len(self.images)):
            img_data = self.images[i_img].get_data()
            img_data = np.pad(img_data, ((self.offset[0], self.offset[0]), (self.offset[1], self.offset[1]), (self.offset[2], self.offset[2])),'constant', constant_values=(0, 0))
            self.images[i_img] = nib.Nifti1Image(img_data, self.images[i_img].affine, self.images[i_img].header, file_map=self.images[i_img].file_map)

    def setup_intensity(self):
        # TODO: change for segmentation images
        for i, image in enumerate(self.images):
            if str(i) in self.im_params.images_parameters:
                vmin = self.im_params.images_parameters[str(i)].vmin
                vmax = self.im_params.images_parameters[str(i)].vmax
                vmean = self.im_params.images_parameters[str(i)].vmean
                if self.im_params.images_parameters[str(i)].vmode == 'percentile':
                    flattened_volume = image.flatten()
                    first_percentile = np.percentile(flattened_volume[flattened_volume > 0], int(vmin))
                    last_percentile = np.percentile(flattened_volume[flattened_volume > 0], int(vmax))
                    mean_intensity = np.percentile(flattened_volume[flattened_volume > 0], int(vmean))
                    std_intensity = last_percentile - first_percentile
                elif self.im_params.images_parameters[str(i)].vmode == 'mean-std':
                    mean_intensity = (float(vmax) + float(vmin)) / 2.0
                    std_intensity = (float(vmax) - float(vmin)) / 2.0

            else:
                flattened_volume = image.flatten()
                first_percentile = np.percentile(flattened_volume[flattened_volume > 0], 0)
                last_percentile = np.percentile(flattened_volume[flattened_volume > 0], 99)
                mean_intensity = np.percentile(flattened_volume[flattened_volume > 0], 98)
                std_intensity = last_percentile - first_percentile

            self.mean_intensity.append(mean_intensity)
            self.std_intensity.append(std_intensity)

            min_intensity = mean_intensity - std_intensity
            max_intensity = mean_intensity + std_intensity

            for window in self.windows:
                window.figs[i].set_clim(min_intensity, max_intensity)

    def is_point_in_image(self, target_point):
        return 0 <= target_point.x < self.image_dim[0] and 0 <= target_point.y < self.image_dim[1] and 0 <= target_point.z < self.image_dim[2]

    def change_intensity(self, event, plot=None):
        if abs(event.xdata - self.press[0]) < 1 and abs(event.ydata - self.press[1]) < 1:
            self.press = event.xdata, event.ydata
            return

        if time() - self.last_update <= self.update_freq:
            return

        self.last_update = time()

        xlim, ylim = self.windows[0].axes.get_xlim(), self.windows[0].axes.get_ylim()
        mean_intensity_factor = (event.xdata - xlim[0]) / float(xlim[1] - xlim[0])
        std_intensity_factor = (event.ydata - ylim[1]) / float(ylim[0] - ylim[1])
        mean_factor = self.mean_intensity[0] - (mean_intensity_factor - 0.5) * self.mean_intensity[0] * 3.0
        std_factor = self.std_intensity[0] + (std_intensity_factor - 0.5) * self.std_intensity[0] * 2.0
        min_intensity = mean_factor - std_factor
        max_intensity = mean_factor + std_factor

        for window in self.windows:
            window.change_intensity(min_intensity, max_intensity)

    def get_event_coordinates(self, event, plot=None):
        point = None
        if plot.view == 1:
            point = Coordinate([self.current_point.x,
                                int(round(event.ydata)),
                                int(round(event.xdata)), 1])
        elif plot.view == 2:
            point = Coordinate([int(round(event.ydata)),
                                self.current_point.y,
                                int(round(event.xdata)), 1])
        elif plot.view == 3:
            point = Coordinate([int(round(event.ydata)),
                                int(round(event.xdata)),
                                self.current_point.z, 1])
        elif plot.view == 4:
            xyz_coord = np.array([int(round(event.xdata)), int(round(event.ydata)), int(self.get_event_slice(event, plot))])
            point = Coordinate([xyz_coord[0],
                                xyz_coord[1],
                                xyz_coord[2], self.images[0].get_data()[tuple(xyz_coord+self.offset)]])
        return point

    def get_event_slice(self, event, plot=None):

        return np.ravel_multi_index((event.inaxes.rowNum, event.inaxes.colNum), dims=(event.inaxes.numRows, event.inaxes.numCols))

    def draw(self):
        for window in self.windows:
            window.fig.figure.canvas.draw()

    def start(self):
        plt.show()


class AxialViewer(Viewer):
    """
    This class is a visualizer for volumes (3D images). It displays the image as axial slices.
    Assumes AIL orientation
    """
    def __init__(self, list_images, plotdir, visualization_parameters=None):
        self.plotdir = plotdir
        self.fit_fig = plt.figure('IVIM fit: '+os.path.abspath(list_images[0].file_map['image'].filename), figsize=(10, 8))
        self.fit_ax = self.fit_fig.subplots()

        if isinstance(list_images, nib.Nifti1Image):
            list_images = [list_images]
        if not visualization_parameters:
            visualization_parameters = ParamMultiImageVisualization([ParamImageVisualization()])
        super(AxialViewer, self).__init__(list_images, visualization_parameters)

        self.compute_offset()
        self.pad_data()

        self.current_point = Coordinate([int(self.images[0].shape[0] / 2), int(self.images[0].shape[1] / 2), int(self.images[0].shape[2] / 2)])

        """Set subplots according to the number of slices in the image"""
        n_slices = self.image_dim[2]
        n_rows_subplot = np.ceil(n_slices / 4.)
        n_cols_subplot = np.ceil(n_slices/n_rows_subplot)
        for i_subplot in range(n_slices):
            ax = self.fig.add_subplot(n_rows_subplot, n_cols_subplot, i_subplot+1)
            self.windows.append(SinglePlot(ax=ax, images=self.images, viewer=self, view=4, im_params=visualization_parameters, display_cross=''))
            self.windows[i_subplot].update_slice(self.offset[2] + i_subplot)
        self.fig.suptitle(self.images[0].file_map['image'].filename)
        cbar_ax = self.fig.add_subplot(n_rows_subplot*10, 1, 1)
        cb = self.fig.colorbar(self.windows[0].figs[0], cax=cbar_ax, orientation='horizontal')
        cb.ax.xaxis.set_ticks_position('top')
        cb.ax.xaxis.set_label_position('top')

        """Connect buttons to user actions"""
        for window in self.windows:
            window.connect()

        # if no intensity bounds ask by user, compute relevant intensity bounds to display
        if not visualization_parameters.images_parameters['0'].clim:
            self.setup_intensity()

    def move(self, event, plot):
        is_in_axes = False
        for window in self.windows:
            if event.inaxes == window.axes:
                is_in_axes = True
        if not is_in_axes:
            return

        if event.xdata and abs(event.xdata - self.press[0]) < 0.5 and abs(event.ydata - self.press[1]) < 0.5:
            self.press = event.xdata, event.ydata
            return

        if time() - self.last_update <= self.update_freq:
            return

        self.last_update = time()
        self.current_point = self.get_event_coordinates(event, plot)
        point = [self.current_point.x, self.current_point.y, self.current_point.z]
        for window in self.windows:
            if window is plot:
                window.update_slice(point, data_update=False)
            else:
                window.update_slice(point, data_update=True)

        self.press = event.xdata, event.ydata
        return

    def on_press(self, event, plot=None):
        if event.button == 1:
            event.xdata -= self.offset[0]
            event.ydata -= self.offset[1]
            self.display_fit_plot(event, plot)
        else:
            return

    def on_motion(self, event, plot=None):
        if event.button == 1:
            return #self.move(event, plot)
        else:
            return

    def on_release(self, event, plot=None):
        if event.button == 1:
            return
        else:
            return

    def display_fit_plot(self, event, plot=None):
        # self.fit_ax.cla()

        vox_coord = self.get_event_coordinates(event, plot)
        print('Selected voxel: x='+str(vox_coord.x)+', y='+str(vox_coord.y)+', z='+str(vox_coord.z)+', value='+str(vox_coord.value))
        # # load PNG plot from Matlab fit then add 1 to all coordinates as Matlab coordinates start at 1 instead of 0 (as Python)
        # vox_fit_plot = mpimg.imread(self.plotdir+'/z'+str(vox_coord.z+1)+'_y'+str(vox_coord.y+1)+'_x'+str(self.image_dim[0]-(vox_coord.x+1-1))+'.png')  # in last data X-axis is reversed due to data loading in Matlab (function load_nii_data has been modified since then)
        # load PNG plot from Python fit (starting at 0)
        vox_fit_plot = mpimg.imread(self.plotdir+'/z'+str(vox_coord.z)+'_y'+str(vox_coord.y)+'_x'+str(vox_coord.x)+'.png')  # in last data X-axis is reversed due to data loading in Matlab (function load_nii_data has been modified since then)
        self.fit_ax.imshow(vox_fit_plot)
        self.fit_ax.axis('off')
        self.fit_fig.canvas.draw()

        return


class ParamImageVisualization(object):
    def __init__(self, id='0', mode='image', cmap='gray', interp='nearest', vmin='0', vmax='99', vmean='98', vmode='percentile', alpha='1.0'):
        self.id = id
        self.mode = mode
        self.cmap = cmap
        self.interp = interp
        self.vmin = vmin
        self.vmax = vmax
        self.vmean = vmean
        self.vmode = vmode
        self.alpha = alpha

    def update(self, params):
        list_objects = params.split(',')
        for obj in list_objects:
            if len(obj) < 2:
                sys.exit('Please check parameter -param (usage changed from previous version)')
            objs = obj.split('=')
            setattr(self, objs[0], objs[1])

class ParamMultiImageVisualization(object):
    """
    This class contains a dictionary with the params of multiple images visualization
    """
    def __init__(self, list_param):
        self.ids = []
        self.images_parameters = dict()
        for param_image in list_param:
            if isinstance(param_image, ParamImageVisualization):
                self.images_parameters[param_image.id] = param_image
            else:
                self.addImage(param_image)

    def addImage(self, param_image):
        param_im = ParamImageVisualization()
        param_im.update(param_image)
        if param_im.id != 0:
            if param_im.id in self.images_parameters:
                self.images_parameters[param_im.id].update(param_image)
            else:
                self.images_parameters[param_im.id] = param_im
        else:
            sys.exit("ERROR: parameters must contain 'id'")


class Coordinate():
    def __init__(self, xyzval_arr):
        self.x = xyzval_arr[0]
        self.y = xyzval_arr[1]
        self.z = xyzval_arr[2]
        if len(xyzval_arr) == 4:
            self.value = xyzval_arr[3]
        else:
            self.value = 0


# Start program
if __name__ == "__main__":

    # parse arguments
    parser = argparse.ArgumentParser(description='Displays parameter map and enables user to inspect fitting by displaying the fit plot to any voxel he/she clicks on.', formatter_class=argparse.RawTextHelpFormatter)

    optionalArgs = parser._action_groups.pop()
    requiredArgs = parser.add_argument_group('required arguments')

    requiredArgs.add_argument('-i', dest='fname_image', help='Image to display.', type=str, required=True)
    requiredArgs.add_argument('-plotdir', dest='plot_dir', help='Path to folder where fit plots are stored.', type=str, required=True)

    optionalArgs.add_argument('-param', dest='display_param', help='Display parameters for the image. Separate parameters with \":\" and assign their value with \"=\".'
                                  '\n\t- cmap: image colormap (e.g. jet, gray, see https://matplotlib.org/3.1.1/gallery/color/colormap_\nreference.html for all colormaps)'
                                  '\n\t- clim: image intensity boundaries (to be separated with \";\", becareful if you type the command \nin a Terminal, you will have to add a \"\\\" before it to avoid the Terminal to interpret it, i.e. \"\\;\")',
                              type=str, required=False, default='params_map')
    parser._action_groups.append(optionalArgs)

    args = parser.parse_args()

    # print citation
    print '\n\n'
    print '\n****************************** <3 Thank you for using our toolbox! <3 *********************************' \
          '\n********************************* PLEASE CITE THE FOLLOWING PAPER *************************************' \
          '\nLévy S., Rapacchi S., Massire A., Troalen T., Feiweier T., Guye M., Callot V., Intra-Voxel Incoherent ' \
          '\nMotion at 7 Tesla to quantify human spinal cord microperfusion: limitations and promises, Magnetic ' \
          '\nResonance in Medicine, 1902:334-357, 2019.' \
          '\n*******************************************************************************************************'
    print '\n\n'

    # load images
    list_images = [nib.load(fname) for fname in [args.fname_image]]

    # load display parameters for each image
    param_image1 = ParamImageVisualization()
    visualization_parameters = ParamMultiImageVisualization([param_image1])
    param_images = [args.display_param]
    # update registration parameters
    for param in param_images:
        visualization_parameters.addImage(param)

    # all slices in axial view are displayed
    viewer = AxialViewer(list_images, args.plot_dir, visualization_parameters)
    viewer.start()
