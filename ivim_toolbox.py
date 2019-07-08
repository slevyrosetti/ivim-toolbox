#!/usr/bin/env python2

"""Graphical User Interface for ivim-toolbox.

Created on Tue Jul  2 17:45:43 2019

@author: slevyrosetti
"""

import wx
import os
import ivim_fitting


class MyFrameName(wx.Frame):
    def __init__(self, parent, id, title):
        wx.Frame.__init__(self, parent, id=id, title=title, size=(500, 500))

        fgrid = wx.FlexGridSizer(6, 2, 5, 10)

        # load path buttons
        self.paths = ["None", "None", "None", "None"]
        self.LoadpathDWIbutton = wx.Button(self, -1, "4D diffusion-weighted images (nifti file)")
        self.LoadpathDWIbutton.Bind(wx.EVT_BUTTON, lambda evt:  self.Loadpath(evt, 0))

        self.LoadpathBvalbutton = wx.Button(self, -1, "B-value file (text file)")
        self.LoadpathBvalbutton.Bind(wx.EVT_BUTTON, lambda evt: self.Loadpath(evt, 1))

        self.LoadpathMaskbutton = wx.Button(self, -1, "Binary mask defining voxels to be fitted (nifti file)")
        self.LoadpathMaskbutton.Bind(wx.EVT_BUTTON, lambda evt: self.Loadpath(evt, 2))

        self.p0 = wx.TextCtrl(self, value=self.paths[0], style=wx.TE_READONLY | wx.TE_CENTER)
        self.p1 = wx.TextCtrl(self, value=self.paths[1], style=wx.TE_READONLY | wx.TE_CENTER)
        self.p2 = wx.TextCtrl(self, value=self.paths[2], style=wx.TE_READONLY | wx.TE_CENTER)

        # model list
        ModelChoiceName = wx.StaticText(self, label='Fit approach', style=wx.ALIGN_LEFT)
        self.ModelChoice = wx.Choice(self, choices=['Two-step', 'One-step'])
        self.ModelChoice.SetSelection(1)

        # output folder
        self.LoadpathOFolderbutton = wx.Button(self, -1, "Output folder (need to exist)")
        self.LoadpathOFolderbutton.Bind(wx.EVT_BUTTON, lambda evt: self.Loadpath(evt, 3, "dir"))
        self.p3 = wx.TextCtrl(self, value=self.paths[3], style=wx.TE_READONLY | wx.TE_CENTER)

        # multi-threading checkbox
        self.MTCheckBox = wx.CheckBox(self, label='Multi-threading')
        self.MTCheckBox.SetValue(True)

        # Run button
        self.RunButton = wx.Button(self, -1, "Run fit")
        self.RunButton.Bind(wx.EVT_BUTTON, self.ClickedOnRun)

        # adapt path display to window size (DO NOT WORK APPARENTLY)
        fgrid.AddGrowableCol(1, 1)
        fgrid.AddMany([(self.LoadpathDWIbutton), (self.p0, wx.EXPAND), (self.LoadpathBvalbutton), (self.p1, wx.EXPAND), (self.LoadpathMaskbutton), (self.p2, wx.EXPAND), (ModelChoiceName),
                       (self.ModelChoice), (self.LoadpathOFolderbutton), (self.p3, wx.EXPAND), (self.MTCheckBox), (self.RunButton)])

        self.SetSizer(fgrid)
        self.Fit()
        self.Centre()

    def Loadpath(self, event, index, itemType="file"):

        if itemType == "file":
            with wx.FileDialog(self, "Export Result ", style=wx.FD_OPEN) as fileDialog:

                if fileDialog.ShowModal() == wx.ID_CANCEL:
                    return  # the user changed their mind
                pathname = fileDialog.GetPath()

        else:
            with wx.DirDialog(self, "Export Result ", style=wx.FD_OPEN) as dirDialog:

                if dirDialog.ShowModal() == wx.ID_CANCEL:
                    return  # the user changed their mind
                pathname = dirDialog.GetPath()

        # save the current contents in the file
        self.paths[index] = pathname
        p=[self.p0, self.p1, self.p2, self.p3]
        p[index].SetValue(pathname)


    def ClickedOnRun(self, event):

        print "\n\n\n======= PARAMETERS ======="
        print "4D diffusion-weighted file: "+self.paths[0]
        print "B-value file: "+self.paths[1]
        print "Mask file: "+self.paths[2]

        model_chosen_index = self.ModelChoice.GetSelection()
        if model_chosen_index == -1:
            return
        model_chosen = self.ModelChoice.GetString(model_chosen_index)

        print "Fit approach: "+model_chosen

        print "Output folder: "+self.paths[3]

        MTcheckbox_value = self.MTCheckBox.GetValue()
        print "Multi-threading: "+str(MTcheckbox_value)
        print "==========================\n\n"

        # run fit
        ivim_fitting.main(dwi_fname=self.paths[0], bval_fname=self.paths[1], mask_fname=self.paths[2], model=model_chosen, ofolder=self.paths[3], multithreading=MTcheckbox_value)



if __name__=='__main__':
    app = wx.PySimpleApp()
    frame = MyFrameName(parent=None, id=-1, title='IVIM toolbox')
    frame.Show()
    app.MainLoop()