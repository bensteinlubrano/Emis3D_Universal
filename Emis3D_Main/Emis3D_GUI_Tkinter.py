#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Apr 11 18:17:46 2023

@author: br0148
"""

import tkinter as tk
from tkinter import ttk
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
import numpy as np

class Emis3D_GUI(object):
    
    def __init__(self):
    
        print("loading...")
        
        self.window = tk.Tk()
        self.window.title('Emis3D')
        self.window.geometry("1800x800")
        
        style = ttk.Style(self.window)
        style.theme_use('clam')
        
        tabControl = ttk.Notebook(self.window)
        
        oneTimestepTab = ttk.Frame(tabControl)
        tabControl.add(oneTimestepTab, text='One Timestep Plots')
        
        RadPowerTab = ttk.Frame(tabControl)
        tabControl.add(RadPowerTab, text='Rad Power Overview')
        
        BuildDistsTab = ttk.Frame(tabControl)
        tabControl.add(BuildDistsTab, text="Build RadDists")
        
        tabControl.pack(expand=1, fill="both")
        
        self.one_timestep(Tab = oneTimestepTab)
        self.rad_power_overview(Tab = RadPowerTab)
        self.build_radDists(Tab = BuildDistsTab)
        
        self.window.mainloop()
        
    def build_radDists(self, Tab):
        
        typeOptionsFrame = ttk.Frame(Tab)
        typeOptionsFrame.grid(column=0, row=0, sticky="NW")
        
        buildOptionsFrame = ttk.Frame(Tab)
        buildOptionsFrame.grid(column=0, row=1, sticky="NW")
        
        errorFrame = ttk.Frame(Tab)
        errorFrame.grid(column=0, row=2, sticky="NW")
        
        self.build_radDists_error = ttk.Label(errorFrame, text="")
        self.build_radDists_error.grid(column=0, row=0, padx=100, pady=10)
        
        def build_radDists_error_message(ErrContainer, Message):
            self.build_radDists_error.grid_forget()
            self.build_radDists_error = ttk.Label(ErrContainer, text=Message)
            self.build_radDists_error.grid(column=0, row=0, padx=100, pady=10)
        
        radDistType_prompt = ttk.Label(typeOptionsFrame, text="Choose RadDist type")
        radDistType_prompt.grid(column=0, row=0, padx=30, pady=10)
        
        radDistTypeOptions = ["Helical", "Toroidal", "M3DC1"]
        buildRadDistType = tk.StringVar(master=typeOptionsFrame)
        buildRadDistType.set(radDistTypeOptions[0])
        radDistMenu = tk.OptionMenu(typeOptionsFrame, buildRadDistType, *radDistTypeOptions)
        radDistMenu.grid(column=0, row=1, padx=30, pady=10)
        
        tokamakType_prompt = ttk.Label(typeOptionsFrame, text="Choose Tokamak")
        tokamakType_prompt.grid(column=1, row=0, padx=30, pady=10)
        
        tokamakOptions = ["SPARC", "JET"]
        tokamakName = tk.StringVar(master=typeOptionsFrame)
        tokamakName.set(tokamakOptions[0])
        tokamakMenu = tk.OptionMenu(typeOptionsFrame, tokamakName, *tokamakOptions)
        tokamakMenu.grid(column=1, row=1, padx=30, pady=10)
        
        chooseTypeButton = ttk.Button(typeOptionsFrame, text='Enter',\
            command=lambda:build_options(Container=buildOptionsFrame,\
                                         ErrContainer=errorFrame,\
                                         RadDistType=buildRadDistType.get()))
        chooseTypeButton.grid(column=2, row=1, ipadx=10, ipady=3)
        
        def build_options(Container, ErrContainer, RadDistType):
            
            build_radDists_error_message(ErrContainer, "Loading")
                
            if RadDistType=="Helical" or RadDistType=="M3DC1":
                build_radDists_error_message(ErrContainer,\
                                             Message=RadDistType + "s are not yet configured to build from GUI")
                return 1
            elif RadDistType=="Toroidal":
                eqfile=None
            
            try:
                if tokamakName.get()=="SPARC":
                    from RadDist import SPARC
                    tokamak = SPARC(Mode="Build", Reflections="False", Eqfile = eqfile)
                elif tokamakName.get()=="JET":
                    from RadDist import SPARC
                    tokamak = JET(Mode="Build", Reflections="False", Eqfile = eqfile)
                    
                build_radDists_error_message(ErrContainer,\
                                             Message="Tokamak object built successfully")
            except:
                build_radDists_error_message(ErrContainer,\
                                             Message="Tokamak object could not be built")
                return 1
            
        
    def one_timestep(self, Tab):
        
        controlFrame = ttk.Frame(Tab)
        plotsFrame = ttk.Frame(Tab)
        
        controlFrame.grid(column=0, row=0, sticky="NW")
        plotsFrame.grid(column=0, row=1, sticky="NW")
        
        if self.comparingTo == "Experiment":
            shotnumber_prompt = ttk.Label(controlFrame, text="Enter Shot Number")
            shotnumber_prompt.grid(column=0, row=0, padx=30, pady=10)
            
            self.shotnumberEntryOneTimestep = ttk.Entry(controlFrame)
            self.shotnumberEntryOneTimestep.grid(column=0, row=1, padx=30, pady=10)
            self.shotnumberEntryOneTimestep.insert(tk.END, '95709')
        
        etime_prompt = ttk.Label(controlFrame, text="Enter Evaluation Time")
        etime_prompt.grid(column=1, row=0, padx=30, pady=10)
        
        self.etimeEntry = ttk.Entry(controlFrame)
        self.etimeEntry.grid(column=1, row=1, padx=30, pady=10)
        if self.comparingTo == "Experiment":
            self.etimeEntry.insert(tk.END, '50.93')
        elif self.comparingTo == "Simulation":
            self.etimeEntry.insert(tk.END, '0')
        
        type1_prompt = ttk.Label(controlFrame, text="Enter RadDist Type 1")
        type1_prompt.grid(column=2, row=0, padx=30, pady=10)
        
        self.type1entry = ttk.Entry(controlFrame)
        self.type1entry.grid(column=2, row=1, padx=30, pady=10)
        self.type1entry.insert(tk.END, 'Helical')
        
        type2_prompt = ttk.Label(controlFrame, text="Enter RadDist Type 2")
        type2_prompt.grid(column=3, row=0, padx=30, pady=10)
        
        self.type2entry = ttk.Entry(controlFrame)
        self.type2entry.grid(column=3, row=1, padx=30, pady=10)
        self.type2entry.insert(tk.END, 'Toroidal')
        
        move_peak_prompt = ttk.Label(controlFrame, text="Allow Peak Movement")
        move_peak_prompt.grid(row=0, column=4, padx=30, pady=10)
        
        self.movePeakToggle1 = tk.Button(controlFrame, text="No", width=12, relief="raised")
        def peakToggle():
            if self.movePeakToggle1.config('relief')[-1] == 'sunken':
                self.movePeakToggle1.config(text="No")
                self.movePeakToggle1.config(relief="raised")
            else:
                self.movePeakToggle1.config(text="Yes")
                self.movePeakToggle1.config(relief="sunken")
        self.movePeakToggle1.config(command=peakToggle)
        self.movePeakToggle1.grid(row=1, column=4, padx=30, pady=10)
        
        buttonRefreshTimestep = ttk.Button(controlFrame, text='Enter',\
            command=lambda:self.refresh_timestep(Container=plotsFrame))
        buttonRefreshTimestep.grid(column=5, row=1, ipadx=10, ipady=3)
        
    def rad_power_overview(self, Tab):
        
        controlFrame = ttk.Frame(Tab)
        controlFrame2 = ttk.Frame(Tab)
        plotsFrame = ttk.Frame(Tab)
        resultsFrame = ttk.Frame(Tab)
        
        controlFrame.grid(column=0, row=0, sticky="NW")
        controlFrame2.grid(column=0, row=1, sticky="NW")
        resultsFrame.grid(column=1, row=0, sticky="NW")
        plotsFrame.grid(column=0, row=2, sticky="NW")
        
        if self.comparingTo == "Experiment":
            shotnumber_prompt = ttk.Label(controlFrame, text="Enter Shot Number")
            shotnumber_prompt.grid(column=0, row=0, padx=30, pady=10)
            
            self.shotnumberEntryRadPowerOverview = ttk.Entry(controlFrame)
            self.shotnumberEntryRadPowerOverview.grid(column=0, row=1, padx=30, pady=10)
            self.shotnumberEntryRadPowerOverview.insert(tk.END, '95709')
        
            start_time_prompt = ttk.Label(controlFrame, text="Enter Start Time")
            start_time_prompt.grid(row=0, column=1, padx=30, pady=10)
            
            self.startTimeEntry = ttk.Entry(controlFrame)
            self.startTimeEntry.grid(row=1, column=1, padx=30, pady=10)
            self.startTimeEntry.insert(tk.END, '50.947')
            
            end_time_prompt = ttk.Label(controlFrame, text="Enter End Time")
            end_time_prompt.grid(row=0, column=2, padx=30, pady=10)
            
            self.endTimeEntry = ttk.Entry(controlFrame)
            self.endTimeEntry.grid(row=1, column=2, padx=30, pady=10)
            self.endTimeEntry.insert(tk.END, '50.973')
        
            num_times_prompt = ttk.Label(controlFrame, text="Enter Number of Timesteps")
            num_times_prompt.grid(row=0, column=3, padx=30, pady=10)
            
            self.numTimesEntry = ttk.Entry(controlFrame)
            self.numTimesEntry.grid(row=1, column=3, padx=30, pady=10)
            self.numTimesEntry.insert(tk.END, '53')
        
        pval_mult_prompt = ttk.Label(controlFrame, text="Enter P Value")
        pval_mult_prompt.grid(row=0, column=4, padx=30, pady=10)
        
        self.pvalMultEntry = ttk.Entry(controlFrame)
        self.pvalMultEntry.grid(row=1, column=4, padx=30, pady=10)
        self.pvalMultEntry.insert(tk.END, '0.6827')
        
        make_movie_prompt = ttk.Label(controlFrame2, text="Build Best Fits Movie")
        make_movie_prompt.grid(row=0, column=0, padx=30, pady=10)
        
        self.makeMovieToggle = tk.Button(controlFrame2, text="No", width=12, relief="raised")
        def movieToggle():
            if self.makeMovieToggle.config('relief')[-1] == 'sunken':
                self.makeMovieToggle.config(text="No")
                self.makeMovieToggle.config(relief="raised")
            else:
                self.makeMovieToggle.config(text="Yes")
                self.makeMovieToggle.config(relief="sunken")
        self.makeMovieToggle.config(command=movieToggle)
        self.makeMovieToggle.grid(row=1, column=0, padx=30, pady=10)
        
        make_crossSecs_prompt = ttk.Label(controlFrame2, text="Build Cross Sections\nMovie")
        make_crossSecs_prompt.grid(row=0, column=1, padx=30, pady=5)
        
        self.makeCrossSecsToggle = tk.Button(controlFrame2, text="No", width=12, relief="raised")
        def crossSecsToggle():
            if self.makeCrossSecsToggle.config('relief')[-1] == 'sunken':
                self.makeCrossSecsToggle.config(text="No")
                self.makeCrossSecsToggle.config(relief="raised")
            else:
                self.makeCrossSecsToggle.config(text="Yes")
                self.makeCrossSecsToggle.config(relief="sunken")
        self.makeCrossSecsToggle.config(command=crossSecsToggle)
        self.makeCrossSecsToggle.grid(row=1, column=1, padx=30, pady=10)
        
        move_peak_prompt = ttk.Label(controlFrame2, text="Allow Peak Movement")
        move_peak_prompt.grid(row=0, column=2, padx=30, pady=10)
        
        self.movePeakToggle2 = tk.Button(controlFrame2, text="No", width=12, relief="raised")
        def peakToggle():
            if self.movePeakToggle2.config('relief')[-1] == 'sunken':
                self.movePeakToggle2.config(text="No")
                self.movePeakToggle2.config(relief="raised")
            else:
                self.movePeakToggle2.config(text="Yes")
                self.movePeakToggle2.config(relief="sunken")
        self.movePeakToggle2.config(command=peakToggle)
        self.movePeakToggle2.grid(row=1, column=2, padx=30, pady=10)
        
        make_contours_prompt = ttk.Label(controlFrame2, text="Build KB5 Contour Plots")
        make_contours_prompt.grid(row=0, column=3, padx=30, pady=10)
        
        self.makeBoloContoursToggle = tk.Button(controlFrame2, text="No", width=12, relief="raised")
        def boloContoursToggle():
            if self.makeBoloContoursToggle.config('relief')[-1] == 'sunken':
                self.makeBoloContoursToggle.config(text="No")
                self.makeBoloContoursToggle.config(relief="raised")
            else:
                self.makeBoloContoursToggle.config(text="Yes")
                self.makeBoloContoursToggle.config(relief="sunken")
        self.makeBoloContoursToggle.config(command=boloContoursToggle)
        self.makeBoloContoursToggle.grid(row=1, column=3, padx=30, pady=10)
        
        buttonTimestepRange = ttk.Button(master=controlFrame2, text='Enter',\
            command=lambda:self.display_tot_rad_power(PlotsContainer=plotsFrame, ResultsContainer = resultsFrame))
        buttonTimestepRange.grid(row=1, column=4, ipadx=10, ipady=3)
        
        self.totRadPowerLabel = ttk.Label(resultsFrame, text="Total Radiated Power =")
        self.totRadPowerLabel.grid(row=0, column=0, padx=50, pady=5)
        
        self.lowerBoundLabel = ttk.Label(resultsFrame, text="Lower Bound =")
        self.lowerBoundLabel.grid(row=1, column=0, padx=50, pady=5)
        
        self.upperBoundLabel = ttk.Label(resultsFrame, text="Upper Bound =")
        self.upperBoundLabel.grid(row=2, column=0, padx=50, pady=5)
        
    def init_emis3D_oneTimestep(self):
        
        if self.comparingTo == "Experiment":
            shotnum = int(self.shotnumberEntryOneTimestep.get())
            self.emis = self.init_emis3D_experimental(Shotnumber=shotnum)
        elif self.comparingTo == "Simulation":
            self.emis = self.init_emis3D_simulational()
        
    def refresh_timestep(self, Container):
        self.init_emis3D_oneTimestep()
        self.plots_one_timestep(Container=Container)
        
    def plots_one_timestep(self, Container):
        if self.comparingTo == "Experiment":
            etime = float(self.etimeEntry.get())
        elif self.comparingTo == "Simulation":
            etime = int(self.etimeEntry.get())
        self.emis.calc_fits(Etime = etime)
        
        distType1 = str(self.type1entry.get())
        self.display_fits_array_type1(Container=Container, Emis3DObject = self.emis,\
            Etime = etime, PlotDistType=distType1)
        #self.display_powers_array_type1(Container=Container, Emis3DObject = self.emis,\
        #    Etime = etime, PlotDistType=distType1)
        
        distType2 = str(self.type2entry.get())
        self.display_fits_array_type2(Container=Container, Emis3DObject = self.emis,\
            Etime = etime, PlotDistType=distType2)
        self.display_powers_array_type2(Container=Container, Emis3DObject = self.emis,\
            Etime = etime, PlotDistType=distType2)
        
        self.display_fits_channels(Container=Container, Emis3DObject=self.emis, Etime = etime)
        
    def display_powers_array_type1(self, Container, Emis3DObject, Etime = 50.89, PlotDistType = "Helical",\
            Column=1, Row=2):
        
        if self.movePeakToggle1.config('relief')[-1] == 'sunken':
            movePeak=True
        else:
            movePeak=False
        fig = Emis3DObject.plot_powers_array(Etime = Etime,\
            PlotDistType = PlotDistType, MovePeak=movePeak)
        
        try: 
            self.type1PowersCanvas.get_tk_widget().pack_forget()
        except: 
            pass    
        self.type1PowersCanvas = FigureCanvasTkAgg(fig, master = Container)
        self.type1PowersCanvas.draw()
        self.type1PowersCanvas.get_tk_widget().grid(row=Row, column=Column)
        
    def display_powers_array_type2(self, Container, Emis3DObject, Etime = 50.89, PlotDistType = "Toroidal",\
            Column=3, Row=2):
        
        fig = Emis3DObject.plot_powers_array(Etime = Etime, PlotDistType = PlotDistType)
        
        try: 
            self.type2PowersCanvas.get_tk_widget().pack_forget()
        except: 
            pass    
        self.type2PowersCanvas = FigureCanvasTkAgg(fig, master = Container)
        self.type2PowersCanvas.draw()
        self.type2PowersCanvas.get_tk_widget().grid(row=Row, column=Column)
        
    def display_fits_array_type1(self, Container, Emis3DObject, Etime = 50.89, PlotDistType = "Helical",\
            Column=0, Row=2):
        
        fig = Emis3DObject.plot_fits_array(Etime = Etime, PlotDistType = PlotDistType)
        
        try: 
            self.type1FitsCanvas.get_tk_widget().pack_forget()
        except: 
            pass    
        self.type1FitsCanvas = FigureCanvasTkAgg(fig, master = Container)
        self.type1FitsCanvas.draw()
        self.type1FitsCanvas.get_tk_widget().grid(row=Row, column=Column)
        
    def display_fits_array_type2(self, Container, Emis3DObject, Etime = 50.89, PlotDistType = "Toroidal",\
            Column=2, Row=2):
        
        fig = Emis3DObject.plot_fits_array(Etime = Etime, PlotDistType = PlotDistType)
        
        try: 
            self.type2FitsCanvas.get_tk_widget().pack_forget()
        except: 
            pass    
        self.type2FitsCanvas = FigureCanvasTkAgg(fig, master = Container)
        self.type2FitsCanvas.draw()
        self.type2FitsCanvas.get_tk_widget().grid(row=Row, column=Column)
        
    def display_fits_channels(self, Container, Emis3DObject, Etime, Column=4, Row=2):
        
        fig = Emis3DObject.plot_fits_channels(Etime = Etime)
        
        try: 
            self.kb5ChannelsCanvas.get_tk_widget().pack_forget()
        except: 
            pass    
        self.kb5ChannelsCanvas = FigureCanvasTkAgg(fig, master=Container)
        self.kb5ChannelsCanvas.draw()
        self.kb5ChannelsCanvas.get_tk_widget().grid(row=Row, column=Column)
        
    def init_emis3D_radPowerOverview(self):
        
        if self.comparingTo == "Experiment":
            shotnum = int(self.shotnumberEntryRadPowerOverview.get())
            self.emis = self.init_emis3D_experimental(Shotnumber=shotnum)
        elif self.comparingTo == "Simulation":
            self.emis = self.init_emis3D_simulational()
        
    def display_tot_rad_power(self, PlotsContainer, ResultsContainer):
        self.init_emis3D_radPowerOverview()
        pvalCutoff = float(self.pvalMultEntry.get())
        
        if self.movePeakToggle2.config('relief')[-1] == 'sunken':
            movePeak=True
        else:
            movePeak=False
        
        if self.comparingTo == "Experiment":
            startTime = float(self.startTimeEntry.get())
            endTime = float(self.endTimeEntry.get())
            numTimes = int(self.numTimesEntry.get())
            self.emis.calc_tot_rad_from_exp(\
                    StartTime = startTime, EndTime=endTime, NumTimes=numTimes,\
                    ErrorPool=True, PvalCutoff=pvalCutoff, MovePeak=movePeak)
        elif self.comparingTo == "Simulation":
            self.emis.calc_tot_rad_from_sim(ErrorPool=True, PvalCutoff=pvalCutoff, MovePeak=movePeak)
        
        self.totRadPowerLabel.pack_forget()
        self.totRadPowerLabel = ttk.Label(ResultsContainer, text="Total Radiated Energy = " +\
                                         str(np.around(self.emis.totWrad/1e6, decimals=2)) + " [MJ]")
        self.totRadPowerLabel.grid(row=0, column=0, padx=50, pady=5)
        
        self.lowerBoundLabel.pack_forget()
        self.lowerBoundLabel = ttk.Label(ResultsContainer, text="Lower Bound = " +\
                                        str(np.around(self.emis.lowerTotWrad/1e6, decimals=2)) + " [MJ]")
        self.lowerBoundLabel.grid(row=1, column=0, padx=50, pady=5)
        
        self.upperBoundLabel.pack_forget()
        self.upperBoundLabel = ttk.Label(ResultsContainer, text="Upper Bound = " +\
                                        str(np.around(self.emis.upperTotWrad/1e6, decimals=2)) + " [MJ]")
        self.upperBoundLabel.grid(row=2, column=0, padx=50, pady=5)
        
        self.plots_whole_shot(Container=PlotsContainer, Emis3DObject = self.emis, MovePeak=movePeak)
        
    def plots_whole_shot(self, Container, Emis3DObject, MovePeak):
        
        self.display_plot_overview(Container=Container, Emis3DObject = Emis3DObject)
        #self.display_kb1_powers(Container=Container, Emis3DObject = Emis3DObject)
        if self.makeMovieToggle.config('relief')[-1] == 'sunken':
            Emis3DObject.make_unwrapped_movie(MovePeak=MovePeak)
        """
        if self.makeBoloContoursToggle.config('relief')[-1] == 'sunken':
            Emis3DObject.make_best_fits_contour_plots()
        if self.makeCrossSecsToggle.config('relief')[-1] == 'sunken':
            Emis3DObject.make_crossSec_movie(MovePeak=MovePeak)
        """
        
    def display_kb1_powers(self, Container, Emis3DObject):
        
        fig = Emis3DObject.plot_kb1s_whole_shot()
        
        try: 
            self.kb1Canvas.get_tk_widget().pack_forget()
        except: 
            pass    
        self.kb1Canvas = FigureCanvasTkAgg(fig, master = Container)
        self.kb1Canvas.draw()
        self.kb1Canvas.get_tk_widget().grid(row=1, column=2)
        
    def display_plot_overview(self, Container, Emis3DObject):
        fig = Emis3DObject.plot_overview()
        
        try: 
            self.plotOverviewCanvas.get_tk_widget().pack_forget()
        except: 
            pass
        self.plotOverviewCanvas = FigureCanvasTkAgg(fig, master = Container)
        self.plotOverviewCanvas.draw()
        self.plotOverviewCanvas.get_tk_widget().grid(row=1, column=1)