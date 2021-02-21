"""
GUI for viewing detector images

By Dan Porter
Feb 2021
"""

import sys, os
import time
import numpy as np
import matplotlib.pyplot as plt

from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
try:
    from matplotlib.backends.backend_tkagg import NavigationToolbar2TkAgg
except ImportError:
    from matplotlib.backends.backend_tkagg import NavigationToolbar2Tk as NavigationToolbar2TkAgg

if sys.version_info[0] < 3:
    import Tkinter as tk
else:
    import tkinter as tk

from .main import Map

# Fonts
TF = ["Times", 12]  # entry
BF = ["Times", 14]  # Buttons
SF = ["Times New Roman", 14]  # labels
MF = ["Courier", 8]  # fixed distance format
LF = ["Times", 20]  # title Labels
HF = ["Courier", 12]  # Text widgets (big)
# Colours - background
bkg = 'snow'
ety = 'white'
btn = 'azure'  # 'light slate blue'
opt = 'azure'  # 'light slate blue'
btn2 = 'gold'
bkg_ttl = 'white'
# Colours - active
btn_active = 'grey'
opt_active = 'grey'
# Colours - Fonts
txtcol = 'black'
btn_txt = 'black'
ety_txt = 'black'
opt_txt = 'black'
ttl_txt = 'black'


class SheepsGui:
    """
    A standalone GUI window
    """
    _figure_size = [8, 6]

    def __init__(self, width=30, height=50, peak_centre=None, nsheep=20, nwolves=3):
        """Initialise"""
        # Create Tk inter instance
        self.root = tk.Tk()
        self.root.wm_title('Game of Sheeps')
        # self.root.minsize(width=640, height=480)
        self.root.maxsize(width=self.root.winfo_screenwidth(), height=self.root.winfo_screenheight())
        self.root.tk_setPalette(
            background=bkg,
            foreground=txtcol,
            activeBackground=opt_active,
            activeForeground=txtcol)

        # Create map
        self.width = width
        self.height = height
        self.peak_centre = peak_centre
        self.map = Map(width, height, peak_centre, nsheep, nwolves)

        # Create tkinter Frame
        frame = tk.Frame(self.root)
        frame.pack(side=tk.TOP, fill=tk.BOTH, expand=tk.YES)

        # variables
        self.play = tk.BooleanVar(frame, False)
        self.bigstep = tk.IntVar(frame, 10)
        self.mapwidth = tk.IntVar(frame, width)
        self.mapheight = tk.IntVar(frame, height)
        self.nsheep = tk.IntVar(frame, nsheep)
        self.nwolves = tk.IntVar(frame, nwolves)
        self.maptext = tk.StringVar(frame, self.map.__str__())

        # ---title---
        frm = tk.Frame(frame, bg=bkg_ttl)
        frm.pack(fill=tk.X, expand=tk.YES, padx=3, pady=3)

        var = tk.Label(frm, text='Game of Sheeps', font=LF, fg=ttl_txt)
        var.pack(pady=5)

        # ---Mid-section---
        frm = tk.Frame(frame)
        frm.pack(fill=tk.BOTH, expand=tk.YES)

        # MidLeft ---Left Side---
        side = tk.Frame(frm)
        side.pack(side=tk.LEFT, fill=tk.BOTH, expand=tk.YES)

        # MidLeft ---Figure window---
        sec = tk.Frame(side)
        sec.pack(fill=tk.BOTH, expand=tk.YES)

        self.fig = plt.Figure(figsize=self._figure_size, dpi=80)
        self.fig.patch.set_facecolor('w')
        self.ax = self.fig.add_subplot(111)
        self.ax.set_xticklabels([])
        self.ax.set_yticklabels([])
        self.ax.set_xticks([])
        self.ax.set_yticks([])
        self.ax.set_autoscaley_on(True)
        self.ax.set_autoscalex_on(True)
        self.ax.set_frame_on(False)
        self.current_image = self.ax.imshow(self.map.create_image())
        self.ax.set_position([0, 0, 1, 1])

        canvas = FigureCanvasTkAgg(self.fig, sec)
        canvas.get_tk_widget().configure(bg='black')
        canvas.draw()
        canvas.get_tk_widget().pack(fill=tk.BOTH, expand=tk.YES)

        # MidLeft ---Buttons---
        sec = tk.Frame(side)
        sec.pack(expand=tk.YES)

        var = tk.Button(sec, text='Step', font=BF, command=self.but_step,
                        bg=btn, activebackground=btn_active)
        var.pack(side=tk.LEFT)
        var = tk.Button(sec, text='Big Step', font=BF, command=self.but_bigstep,
                        bg=btn, activebackground=btn_active)
        var.pack(side=tk.LEFT)
        var = tk.Button(sec, text='Play', font=BF, command=self.but_play,
                        bg=btn, activebackground=btn_active)
        var.pack(side=tk.LEFT)
        var = tk.Button(sec, text='Pause', font=BF, command=self.but_pause,
                        bg=btn, activebackground=btn_active)
        var.pack(side=tk.LEFT)
        var = tk.Button(sec, text='Restart', font=BF, command=self.but_restart,
                        bg=btn, activebackground=btn_active)
        var.pack(side=tk.LEFT)

        # MidLeft ---More Buttons---
        sec = tk.Frame(side)
        sec.pack(expand=tk.YES, pady=3)

        var = tk.Button(sec, text='Plot Water', font=BF, command=self.btn_water,
                        bg=btn, activebackground=btn_active)
        var.pack(side=tk.LEFT)

        var = tk.Button(sec, text='Plot Food', font=BF, command=self.btn_food,
                        bg=btn, activebackground=btn_active)
        var.pack(side=tk.LEFT)
        var = tk.Button(sec, text='Sheep', font=BF, command=self.btn_sheep,
                        bg=btn, activebackground=btn_active)
        var.pack(side=tk.LEFT)

        var = tk.Button(sec, text='Wolves', font=BF, command=self.btn_wolf,
                        bg=btn, activebackground=btn_active)
        var.pack(side=tk.LEFT)

        # MidRight ---Details window---
        side = tk.Frame(frm)
        side.pack(side=tk.LEFT, fill=tk.BOTH, expand=tk.YES)

        sec = tk.Frame(side)
        sec.pack(fill=tk.X, expand=tk.YES)

        var = tk.Label(sec, text='Map Size  Width:', font=SF)
        var.pack(side=tk.LEFT)
        var = tk.Entry(sec, textvariable=self.mapwidth, font=TF, width=3, bg=ety, fg=ety_txt)
        var.pack(side=tk.LEFT)
        var = tk.Label(sec, text=' Height:', font=SF)
        var.pack(side=tk.LEFT)
        var = tk.Entry(sec, textvariable=self.mapheight, font=TF, width=3, bg=ety, fg=ety_txt)
        var.pack(side=tk.LEFT)

        sec = tk.Frame(side)
        sec.pack(fill=tk.X, expand=tk.YES)

        var = tk.Label(sec, text='Sheep:', font=SF)
        var.pack(side=tk.LEFT)
        var = tk.Entry(sec, textvariable=self.nsheep, font=TF, width=3, bg=ety, fg=ety_txt)
        var.pack(side=tk.LEFT)
        var = tk.Label(sec, text=' Wolves:', font=SF)
        var.pack(side=tk.LEFT)
        var = tk.Entry(sec, textvariable=self.nwolves, font=TF, width=3, bg=ety, fg=ety_txt)
        var.pack(side=tk.LEFT)

        # MidRight ---Details window---
        sec = tk.LabelFrame(side, text='Current Time', relief=tk.RIDGE)
        sec.pack(fill=tk.X, expand=tk.YES, padx=5, pady=5)

        var = tk.Message(sec, textvariable=self.maptext, font=SF)
        var.pack(expand=tk.YES)

        "-------------------------Start Mainloop------------------------------"
        if not hasattr(sys, 'ps1'):
            # If not in interactive mode, start mainloop
            self.root.protocol("WM_DELETE_WINDOW", self.f_exit)
            self.root.mainloop()

    "------------------------------------------------------------------------"
    "--------------------------General Functions-----------------------------"
    "------------------------------------------------------------------------"

    def update(self):
        image = self.map.create_image()
        self.current_image.set_data(image)
        self.maptext.set(self.map.__str__())
        self.fig.canvas.draw()

    def play_continuous(self):
        if self.play.get():
            self.map.time()
            self.update()
            self.root.after(200, self.play_continuous)

    "------------------------------------------------------------------------"
    "---------------------------Button Functions-----------------------------"
    "------------------------------------------------------------------------"

    def but_step(self):
        """Step button"""
        self.play.set(False)
        self.map.time()
        self.update()

    def but_bigstep(self):
        """Step button"""
        self.play.set(False)
        for n in range(self.bigstep.get()):
            self.map.time()
        self.update()

    def but_play(self):
        """Play button"""
        self.play.set(True)
        self.play_continuous()

    def but_pause(self):
        """Pause button"""
        self.play.set(False)

    def but_restart(self):
        """Restart button"""
        self.map = Map(self.width, self.height, self.peak_centre, self.nsheep.get(), self.nwolves.get())
        self.update()

    def btn_water(self):
        """plot watermap"""
        tot_water = np.sum(self.map.watermap)
        wolf_water = np.sum([w.water for w in self.map.wolf_list])
        sheep_water = np.sum([s.water for s in self.map.sheep_list])
        an_water = wolf_water + sheep_water

        plt.figure(figsize=[8, 6], dpi=60)
        ttl = 'Total water: %s\n Cloud water: %s\n Animal water: %s' % (tot_water, self.map.cloud_water, an_water)
        plt.title(ttl, fontsize=18)
        plt.imshow(self.map.watermap, cmap=plt.get_cmap('Blues'))
        plt.colorbar()
        plt.show()

    def btn_food(self):
        """plot foodmap"""
        plt.figure(figsize=[8, 6], dpi=60)
        map_food = np.sum(self.map.foodmap)
        wolf_food = np.sum([w.food for w in self.map.wolf_list])
        sheep_food = np.sum([s.food for s in self.map.sheep_list])
        an_food = wolf_food + sheep_food
        ttl = 'Map Food: %s\n Animal Food: %s' % (map_food, an_food)
        plt.title(ttl, fontsize=18)
        plt.imshow(self.map.foodmap, cmap=plt.get_cmap('Greens'))
        plt.colorbar()
        plt.show()

    def btn_sheep(self):
        """print sheep"""
        for sh in self.map.sheep_list:
            print(sh)

    def btn_wolf(self):
        """print sheep"""
        for wf in self.map.wolf_list:
            print(wf)

    def f_exit(self):
        """Closes the current data window"""
        self.root.destroy()
