# -*- coding: utf-8 -*-
"""
Created on Mon Feb  6 18:07:26 2023

@author: Cerpa Juan Carlos
"""

import import_beh_data as di
import Functions_behaviour as fb

experiment = di.Experiment('C:/Users/Cerpa Juan Carlos/Documents/Labos/Labo Oxford/OneDrive/Labo_Oxford/Experiments/JC04_JC05_JC06/Behaviour')

fb.plot_correct_choices(experiment)
#fb.plot_latencies_center_side(experiment)
#fb.plot_side_breaks(experiment)
#fb.plot_correlation_latency_choice(experiment)