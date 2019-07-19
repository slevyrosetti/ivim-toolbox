#!/usr/bin/env pythonw

"""Graphical User Interface for ivim-toolbox.

Created on Tue Jul  2 17:45:43 2019

@author: slevyrosetti
"""

import wx
import ivim_fitting
import ivim_simu_compute_required_snr


class FrameFitData(wx.Frame):
    def __init__(self, parent, id, title):
        wx.Frame.__init__(self, parent, id=id, title=title, size=(500, 500))

        fgrid = wx.FlexGridSizer(6, 2, 5, 10)

        # load path buttons
        self.paths = ["None", "None", "None", "None"]
        self.LoadpathDWIbutton = wx.Button(self, -1, "4D diffusion-weighted images (nifti file)")
        self.LoadpathDWIbutton.Bind(wx.EVT_BUTTON, lambda evt:  Loadpath(self, evt, 0, itemType="file"))

        self.LoadpathBvalbutton = wx.Button(self, -1, "B-value file (text file)")
        self.LoadpathBvalbutton.Bind(wx.EVT_BUTTON, lambda evt: Loadpath(self, evt, 1, itemType="file"))

        self.LoadpathMaskbutton = wx.Button(self, -1, "Binary mask defining voxels to be fitted (nifti file)")
        self.LoadpathMaskbutton.Bind(wx.EVT_BUTTON, lambda evt: Loadpath(self, evt, 2, itemType="file"))

        self.p0 = wx.TextCtrl(self, value=self.paths[0], style=wx.TE_READONLY | wx.TE_CENTER)
        self.p1 = wx.TextCtrl(self, value=self.paths[1], style=wx.TE_READONLY | wx.TE_CENTER)
        self.p2 = wx.TextCtrl(self, value=self.paths[2], style=wx.TE_READONLY | wx.TE_CENTER)

        # model list
        ModelChoiceName = wx.StaticText(self, label='Fit approach', style=wx.ALIGN_LEFT)
        self.ModelChoice = wx.Choice(self, choices=['two-step', 'one-step'])
        self.ModelChoice.SetSelection(1)

        # output folder
        self.LoadpathOFolderbutton = wx.Button(self, -1, "Output folder (need to exist)")
        self.LoadpathOFolderbutton.Bind(wx.EVT_BUTTON, lambda evt: Loadpath(self, evt, 3, itemType="dir"))
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
        # ivim_fitting.main(dwi_fname=self.paths[0], bval_fname=self.paths[1], mask_fname=self.paths[2], model=model_chosen, ofolder=self.paths[3], multithreading=MTcheckbox_value)


class FrameRequiredSNR(wx.Frame):
    def __init__(self, parent, title):
        super(FrameRequiredSNR, self).__init__(parent, title=title, size=(500, 500))

        # fgrid = wx.FlexGridSizer(7, 2, 5, 10)
        panel = wx.Panel(self)
        vbox = wx.BoxSizer(wx.VERTICAL)

        # Title
        hbox1 = wx.BoxSizer(wx.HORIZONTAL)
        l1 = wx.StaticText(panel, -1, label="Compute required SNR")
        hbox1.Add(l1, 1, wx.EXPAND | wx.ALIGN_CENTER_HORIZONTAL | wx.ALL, 5)
        vbox.Add(hbox1)

        # Fit approach
        hbox2 = wx.BoxSizer(wx.HORIZONTAL)
        # static text
        l2 = wx.StaticText(panel, -1, "Fit approach")
        hbox2.Add(l2, 1, wx.EXPAND | wx.ALIGN_LEFT | wx.ALL, 5)
        # drop down menu
        self.ApproachChoice = wx.Choice(panel, choices=['two-step', 'one-step'])
        self.ApproachChoice.SetSelection(1)
        hbox2.Add(self.ApproachChoice, 1, wx.EXPAND|wx.ALIGN_RIGHT|wx.ALL, 5)
        vbox.Add(hbox2)

        # output folder
        hbox3 = wx.BoxSizer(wx.HORIZONTAL)
        # static text
        l3 = wx.StaticText(panel, -1, "Output folder (need to exist)")
        hbox3.Add(l3, 1, wx.EXPAND | wx.ALIGN_LEFT | wx.ALL, 5)
        # load button
        self.LoadpathOFolderbutton = wx.Button(panel, -1, "Load")
        self.LoadpathOFolderbutton.Bind(wx.EVT_BUTTON, lambda evt: Loadpath(panel, evt, 0, itemType="dir"))
        hbox3.Add(self.LoadpathOFolderbutton, 1, wx.EXPAND | wx.ALIGN_LEFT | wx.ALL, 5)
        # text showing path
        self.paths = ["None"]
        self.p0 = wx.TextCtrl(panel, value=self.paths[0], style=wx.TE_READONLY | wx.TE_CENTER)
        hbox3.Add(self.p0, 1, wx.EXPAND | wx.ALIGN_RIGHT | wx.ALL, 5)
        vbox.Add(hbox3)

        # B-value distribution
        hbox4 = wx.BoxSizer(wx.HORIZONTAL)
        # static text
        l4 = wx.StaticText(panel, -1, "B-value distribution (e.g. 0,10,50,500,800)")
        hbox4.Add(l4, 1, wx.EXPAND | wx.ALIGN_LEFT | wx.ALL, 5)
        # field to write
        self.bval_field = wx.TextCtrl(panel)
        hbox4.Add(self.bval_field, 1, wx.EXPAND | wx.ALIGN_RIGHT | wx.ALL, 5)
        vbox.Add(hbox4)

        # Fivim range
        hbox5 = wx.BoxSizer(wx.HORIZONTAL)
        # static text
        l5 = wx.StaticText(panel, -1, "F values (e.g. 0.1,0.2,0.3 or 0.1:0.01:0.3)")
        hbox5.Add(l5, 1, wx.EXPAND | wx.ALIGN_LEFT | wx.ALL, 5)
        # field to write
        self.F_field = wx.TextCtrl(panel)
        hbox5.Add(self.F_field, 1, wx.EXPAND | wx.ALIGN_RIGHT | wx.ALL, 5)
        vbox.Add(hbox5)

        # Dstar range
        hbox6 = wx.BoxSizer(wx.HORIZONTAL)
        # static text
        l6 = wx.StaticText(panel, -1, "D* values (e.g. 11e-3,3e-3,35e-3 or 3e-3:1e-3:30e-3)")
        hbox6.Add(l6, 1, wx.EXPAND | wx.ALIGN_LEFT | wx.ALL, 5)
        # field to write
        self.Dstar_field = wx.TextCtrl(panel)
        hbox6.Add(self.Dstar_field, 1, wx.EXPAND | wx.ALIGN_RIGHT | wx.ALL, 5)
        vbox.Add(hbox6)

        # D range
        hbox7 = wx.BoxSizer(wx.HORIZONTAL)
        # static text
        l7 = wx.StaticText(panel, -1, "D values (e.g. 0.3e-3,0.4e-3,0.5e-3 or 0.3e-3:0.1e-3:10e-3)")
        hbox7.Add(l7, 1, wx.EXPAND | wx.ALIGN_LEFT | wx.ALL, 5)
        # field to write
        self.D_field = wx.TextCtrl(panel)
        hbox7.Add(self.D_field, 1, wx.EXPAND | wx.ALIGN_RIGHT | wx.ALL, 5)
        vbox.Add(hbox7)

        # Button run calculation
        hbox8 = wx.BoxSizer(wx.HORIZONTAL)
        # static text
        self.RunButton = wx.Button(self, -1, "Run calculation")
        hbox8.Add(self.RunButton, 1, wx.EXPAND | wx.ALIGN_CENTER | wx.ALL, 5)
        vbox.Add(hbox8)
        self.RunButton.Bind(wx.EVT_BUTTON, self.ClickedOnRun)

        panel.SetSizer(vbox)
        panel.SetSize((500, 300))

        self.Centre()
        self.Fit()

    # def OnKeyTyped(self, event):
    #     print event.GetString()
    #
    # def OnEnterPressed(self, event):
    #     print "Enter pressed"
    #
    # def OnMaxLen(self, event):
    #     print "Maximum length reached"
    #
    def ClickedOnRun(self, event):

        # retrieve selected fit approach from drop down menu
        selected_approach_idx = self.ApproachChoice.GetSelection()
        if selected_approach_idx == -1:
            return
        selected_approach = self.ApproachChoice.GetString(selected_approach_idx)

        print "\n\n\n======= PARAMETERS ======="
        print "Fit approach: "+selected_approach
        print "Output folder: "+self.paths[0]
        print "B-value distribution: "+self.bval_field.GetValue()
        print "F true values: "+self.F_field.GetValue()
        print "D* true values: "+self.Dstar_field.GetValue()
        print "D true values: "+self.D_field.GetValue()
        print "==========================\n\n"

        # run SNR calculation
        # ivim_simu_compute_required_snr.main(model=selected_approach, ofolder=self.paths[0], bvals=self.bval_field.GetValue(), condition='FDstar', snr_init=1500., F_range=self.F_field.GetValue(), Dstar_range=self.Dstar_field.GetValue(), D_range=self.D_field.GetValue())


def Loadpath(frame, event, index, itemType="file"):

    if itemType == "file":
        with wx.FileDialog(frame, "Export Result ", style=wx.FD_OPEN) as fileDialog:

            if fileDialog.ShowModal() == wx.ID_CANCEL:
                return  # user changed his mind
            pathname = fileDialog.GetPath()

    else:
        with wx.DirDialog(frame, "Export Result ", style=wx.FD_OPEN) as dirDialog:

            if dirDialog.ShowModal() == wx.ID_CANCEL:
                return  # user changed his mind
            pathname = dirDialog.GetPath()

    # save the current contents in the file
    frame.paths[index] = pathname
    p=[frame.p0, frame.p1, frame.p2, frame.p3]
    p[index].SetValue(pathname)

if __name__=='__main__':
    app = wx.App()
    # instance and show frame to fit data
    frameFitData = FrameFitData(parent=None, id=-1, title='IVIM toolbox: fit data')
    frameFitData.Show()
    # instance and show frame to compute required SNR
    frameRequiredSNR = FrameRequiredSNR(parent=None, title='IVIM toolbox: compute required SNR')
    frameRequiredSNR.Show()
    app.MainLoop()