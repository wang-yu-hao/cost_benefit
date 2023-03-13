# -*- coding: utf-8 -*-
"""
Created on Wed Feb 15 17:27:37 2023

@author: Cerpa Juan Carlos
"""
import os

import import_photo_beh_path as pbd #to get the path to the photometry and behavioural files of interest
import import_beh_data as di #for behavioural files
import sync_pho_data_newMarta as sp #to denoised, correct the photometry signals and synchronize photometry with behaviour system
import Functions_photometry as fp #where the functions for plotting are

#dir_folder = 'C:/Users/juanc/OneDrive/Labo_Oxford/Experiments/'
dir_folder = 'C:/Users/Cerpa Juan Carlos/Documents/Labos/Labo Oxford/OneDrive/Labo_Oxford/Experiments/'
cohort = 'JC04_JC05_JC06/' #optional

dir_folder_session = os.path.join(dir_folder, cohort, 'Behaviour') #folder for behaviour files
dir_folder_pho = os.path.join(dir_folder, cohort, 'Recordings', 'DLS') #folder for recordings files

#all cohorts, GCaMP
all_D2_DLS = ['3.6a', '3.6b', '3.6d', '5.4a', '5.4e', '6.2b', #for DLS
              '7.5b', '7.5d', '7.6a', '8.2c', '8.3b', '8.3c', '8.3d'] 

all_D2_DMS = ['3.6a', '3.6b', '3.6d', '5.4e', '6.2a', '6.2b',  #for DMS
              '7.5b', '7.5d', '7.6a', '7.7d', '8.2c', '8.2d', '8.3b', '8.3c', '8.3d'] 

all_D1_DLS = ['3.3d', '3.5b', '3.5d', '4.4b', '4.4c', '5.2c', '5.2g', '5.2i', '6.3a',
              '6.3b', '6.3c', '6.3d', '6.3f', '8.5b']
all_D1_DMS = ['3.3d', '3.5b', '4.4b', '4.4c', '5.2a', '5.2c', '5.2g', '5.2i', '6.3a',
              '6.3b', '6.3c', '6.3d', '6.3f']

#dLight - recording of Dopamine release in wild type animals
DLS_mice_dlight = ['7.6b', '7.7c', '8.3a', '8.3e', '8.3f'] #for DLS, D2 wt littermates
DMS_mice_dlight = ['7.7c', '8.3e'] #for DMS, D2 wt littermates
mice = ['8.5b'] #or ['8.3e'] D1 WT littermate dLight, D83e DMS ; D85b DLS

mouse = ['3.5b']
day = ['2021-11-15']

all_days = ['2021-11-06', '2021-11-08', '2021-11-09', '2021-11-10', '2021-11-11', '2021-11-12',
           '2021-11-13', '2021-11-15', '2021-11-16', 
           '2021-11-18','2021-11-19', '2021-11-20', '2021-11-21', '2021-11-22', '2021-11-23',
           '2021-11-24', '2021-11-25', '2021-11-26', '2021-11-27', '2021-11-28', '2021-11-29',
           '2021-11-30', '2021-12-01', '2021-12-02', '2021-12-03', '2021-12-04', '2021-12-05',
           '2021-12-06', '2021-12-07', '2021-12-08', '2021-12-09', '2021-12-10',
           '2022-03-18', '2022-03-20', '2022-03-21', '2022-03-22', '2022-03-23', '2022-03-24',
           '2022-03-25', '2022-03-26', '2022-03-27', '2022-03-28', '2022-03-29', '2022-03-30',
           '2022-04-01', '2022-04-02', '2022-04-04', '2022-04-05', '2022-04-06', '2022-04-07',
           '2022-04-12', '2022-04-13', '2022-04-14', '2022-04-15', '2022-04-16', '2022-04-17',
           '2022-04-19', '2022-04-20', '2022-04-21', '2022-04-22', '2022-04-23', '2022-04-25',
           '2022-04-26', '2022-04-27', '2022-04-28', '2022-04-29', '2022-04-30', '2022-05-02',
           '2022-05-03', '2022-05-04', '2022-05-05', '2022-05-06', '2022-05-07', '2022-05-08',
           '2022-05-09', '2022-05-10', '2022-05-11', '2022-05-12', '2022-05-13', '2022-05-14',
           '2022-05-15', '2022-05-16', '2022-05-17', '2022-05-18', '2022-05-19', '2022-05-20',
           '2022-05-21', '2022-05-23', '2022-05-24', '2022-05-25', '2022-05-26', '2022-05-27',
           '2022-05-28', '2022-05-29', '2022-05-30', '2022-05-31', '2022-06-01',
           '2022-11-12', '2022-11-13', '2022-11-14', '2022-11-15', '2022-11-16', '2022-11-18', '2022-11-19',
           '2022-11-20', '2022-11-21', '2022-11-22', '2022-11-23', '2022-11-24', '2022-11-25', '2022-11-26',
           '2022-11-27', '2022-11-28', '2022-11-29', '2022-11-30', '2022-12-01', '2022-12-02', '2022-12-03',
           '2022-12-04', '2022-12-05', '2022-12-06', '2022-12-07', '2022-12-08', '2022-12-09',
           '2022-12-13', '2022-12-14', '2022-12-15', '2022-12-16', '2022-12-17', '2022-12-18'] #days of recordings for all three cohorts

start_str = 'D' #start_str should be either D for D1 animals or A for D2 animals, replace also D in variable sessions_format and photometry_format too
all_sessions_path, all_photometry_path = pbd.import_sessions_photometry_path(
  dir_folder_session, dir_folder_pho, start_str=start_str, sessions_format='D{id}-{datetime}.txt',
  photometry_format='D{id}_{region}_{hemisphere}-{datetime}.ppd', mouse=mouse, day=all_days,
  training='all', region=[], hemisphere=[]) #if recording files for DMS and DLS are in the same folders, use region=['DMS'] or region=['DLS'] to only take recordings from this region. Number of photometry files should match number of behaviour files

sessions = [di.Session(s_path) for s_path in all_sessions_path]

###Denoised and correct signals for photobleaching, New method from Marta (double exponential...)
all_photo_data, all_sample_times_pho, all_sample_times_pyc, all_corrected_signal = \
  zip(*[sp.sync_photometry_data(p_path, 'D{id}_{region}_{hemisphere}', session, low_pass=5, high_pass=0.001)
        for p_path, session in zip(all_photometry_path, sessions)])
  

#gets signals aligned to behaviour and z_scored
t_scale_whole, v_line, trials_to_keep_ids, z_score_signal, scale_latency = fp.get_scaled_photometry_per_trial(sessions, all_sample_times_pyc,
                                                                                           all_corrected_signal, -500, 1000) 

###########################

#enter trial types to plot the corresponding graphs, code currently written to get three curves on the same figure (therefore enter 3 trial types by sublists)
all_trial_type = [['contralateral trial', 'ipsilateral trial', 'up trial']]
#all_trial_type = [['contralateral trial', 'ipsilateral trial', 'up trial'], ['trial_100', 'trial_50', 'trial_20'],
#                   ['trial_100_forced', 'trial_50_forced', 'trial_20_forced'], ['trial_100_free', 'trial_50_free', 'trial_20_free'],
#                   ['trial_100', 'trial_50_rew', 'trial_20_rew'], ['trial_100', 'trial_50_nonrew', 'trial_20_nonrew'],
#                   ['contra_100', 'contra_50', 'contra_20'], ['ipsi_100', 'ipsi_50', 'ipsi_20'], ['up_100', 'up_50', 'up_20'],
#                   ['contra_100_forced', 'contra_50_forced', 'contra_20_forced'], ['contra_100_free', 'contra_50_free', 'contra_20_free'],
#                   ['ipsi_100_forced', 'ipsi_50_forced', 'ipsi_20_forced'], ['ipsi_100_free', 'ipsi_50_free', 'ipsi_20_free'],
#                   ['up_100_forced', 'up_50_forced', 'up_20_forced'], ['up_100_free', 'up_50_free', 'up_20_free']]


#line to plot z_scored photometry signals for the different trial types
fp.plot_photometry(sessions, all_photo_data, z_score_signal, t_scale_whole, v_line, trials_to_keep_ids, all_trial_type, start_str, -500, 1000)


#predictors used for regression
base_predictors = ['contra_choice',
                    'up',
                    'reward',
                    #'cum_rew',
                    'previous_reward',
                    'reward-2',
                    'mean_probas',
                    'variance_probas',
                    #'free_choice_x_correct', 
                    #'contra x centered_mean',
                    #'contra x centered_variance',
                    #'up x centered_mean',
                    #'up x centered_variance',
                    #'forced_trials',
                    #'init_breaks']
                    'forced_trials']
                    #'latency']

#determine the order and what regressors into which subplots
subplots = [['contra_choice', 'up'], ['reward'], ['previous_reward', 'reward-2'],
             #['init_breaks'], #['forced_trials'],
            #['free_choice_x_correct'],
            ['mean_probas', 'variance_probas'],
            #['contra x centered_mean', 'contra x centered_variance'],
            #['up x centered_mean', 'up x centered_variance'],
            ['forced_trials'],
            #['latency'],            
            ['intercept']]

time_start = 500

#plot regressors coefficients also aligned to behavioural events within trial
fp.plot_photometry_regression(sessions, all_photo_data, t_scale_whole, z_score_signal, v_line,
                              time_start, base_predictors, trials_to_keep_ids, scale_latency, forced_choice='only',
                              all_event_lines=True, plot_legend=True, plot_intercept=True,
                              subplots=subplots, plot_correlation=True,
                              type_coef='non_norm', test_visual='dots')

