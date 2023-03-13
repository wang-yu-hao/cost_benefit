# -*- coding: utf-8 -*-
"""
Created on Fri Feb  3 17:04:31 2023

@author: Cerpa Juan Carlos
"""

import import_beh_data as di
import numpy as np
import pylab as plt
import pandas as pd
import seaborn as sns
from scipy import stats
from itertools import zip_longest
import math
import matplotlib.cm as cm
from matplotlib.lines import Line2D

#session = di.Session('C:/Users/juanc/OneDrive/Labo_Oxford/Experiments/JC04_JC05_JC06/Behaviour/A3.6a-2021-11-06-113442.txt')


######################Fraction of good choices during free choice trials##########################

def fraction_funct(choices_pair, high_proba=1):
    good_choice = 0
    for i, x in enumerate(choices_pair):
        if x == high_proba:
            good_choice += 1
    fraction_choices = good_choice / len(choices_pair) if choices_pair else np.nan #calculate the fraction of good choices if data for the pair exist
    return fraction_choices

def fraction_correct_choices(session):
    '''Take free trials
       Extract what was the pair of options and determine if the highest probability has been selected
       Calculate a fraction of good choice'''
    
    all_event_print = session.events_and_print #extract all the events and print lines
    all_prints = [all_event_print[i].name for i in range(len(all_event_print)) if type(all_event_print[i].name) == list and 'T' in all_event_print[i].name[0]] #take only all the print lines appearing at the end of a trial
    trial_type = [all_prints[i][4].split(':')[1] for i in range(len(all_prints))] #all trial type, 'FC' = free trial, 'L' = forced left, 'U'=forced up, 'R'=forced right
    couple = [all_prints[i][5].split(':')[1] for i in range(len(all_prints))] #all couple of choice
    print(session.subject_ID, session.datetime_string)

    proba_choosed = session.trial_data['proba_choosed']
    choices = session.trial_data['choices']  # 1 - left; 2-up; 3- right
    outcomes = session.trial_data['outcomes']  # 1 - rewarded;  0 - non-rewarded 
    free_choice_trials = session.trial_data['free_choice']  # 1 - free choice trial  # 0 - forced choice trial
    free_trial = np.where(free_choice_trials == 1)[0] #trial ids of free trials 
    
    free_proba_choosed = np.array([proba_choosed[x] for x in range(len(proba_choosed)) if x in free_trial]) #proba choosed during free trials
    
    high_prob = [] #store the highest proba in case of a free trial
    low_prob = [] #store the lowest proba in case of a free trial
    for i,x in enumerate(trial_type):
        if x == 'FC': #if trial is a free trial
            high_prob.append(float(all_prints[i][6].split(':')[1]))
            low_prob.append(float(all_prints[i][7].split(':')[1])) 
    
    store_probas = session.store_probas #retrieve the contingencies, e.g [0.5, 1, 0.2] means left = 50%, up=100% and right=20%
    probas = [float(store_probas[i]) for i in range(len(store_probas))] #change it to a list of floats

    choice_100v50 = [] #to store proba selected when 100vs50 is encountered
    choice_100v20 = []
    choice_50v20 = []
    
    for i in range(len(free_trial)):
        if high_prob[i] == 1.0 and low_prob[i] == 0.5:
            choice_100v50.append(free_proba_choosed[i]) 
        elif high_prob[i] == 1.0 and low_prob[i] == 0.2:
            choice_100v20.append(free_proba_choosed[i])
        elif high_prob[i] == 0.5 and low_prob[i] == 0.2:
            choice_50v20.append(free_proba_choosed[i])

    fraction_100v50 = fraction_funct(choice_100v50, high_proba=1)        
    fraction_100v20 = fraction_funct(choice_100v20, high_proba=1)    
    fraction_50v20 = fraction_funct(choice_50v20, high_proba=0.5)
    
    return fraction_100v50, fraction_100v20, fraction_50v20


def average_correct_choices(experiment, subject_IDs='all', when='all'):
    '''average fraction of good choices across animals
    -subjects_IDs can be 'all' or a specified list of subjects
    -same for when
    '''
    
    if subject_IDs == 'all':
        subject_IDs = sorted(experiment.subject_IDs)
    else:
        subject_IDs = sorted([experiment.subject_IDs[i] for i in range(len(experiment.subject_IDs)) if experiment.subject_IDs[i] in subject_IDs])
     
    #subject_IDs = sorted([experiment.subject_IDs[i] for i in range(len(experiment.subject_IDs)) if experiment.subject_IDs[i] in animals])
    subject_sessions = [experiment.get_sessions(subject_ID,when) for i, subject_ID in enumerate(subject_IDs)]
    fraction_individual = [[fraction_correct_choices(session) for session in subject] for subject in subject_sessions]

    sub_average_fraction = [np.nanmean(fraction_individual[i],axis = 0).tolist() for i, subject in enumerate(fraction_individual)] #average for each indiv
    sub_sem_fraction = [stats.sem(fraction_individual[i],axis = 0).tolist() for i, subject in enumerate(fraction_individual)]
    print(subject_IDs)        
    
    mean_fraction_group = np.nanmean(sub_average_fraction,axis=0).tolist() #average across all subjects
    sem_fraction_group = stats.sem(sub_average_fraction,axis=0)
    
    #average data for individuals for all pairs of choices
    sessions_100v50 = [[] for i in range(len(fraction_individual))]
    sessions_100v20 = [[] for i in range(len(fraction_individual))]
    sessions_50v20 = [[] for i in range(len(fraction_individual))]    
    for i in range(len(fraction_individual)):
        for j in range(len(fraction_individual[i])):
            sessions_100v50[i].append(fraction_individual[i][j][0])
            sessions_100v20[i].append(fraction_individual[i][j][1])
            sessions_50v20[i].append(fraction_individual[i][j][2])
           
    sessions = [sessions_100v50, sessions_100v20, sessions_50v20]
    for session_type in sessions: #loop through the different sessions to homogenise the length of sessions
        maxlen = len(max(session_type, key=len))
        for sublist in session_type:
            sublist[:] = sublist + [math.nan] * (maxlen - len(sublist))
    
    return sub_average_fraction, mean_fraction_group, sem_fraction_group, sessions_100v50, sessions_100v20,\
        sessions_50v20

def plot_correct_choices(experiment, subject_IDs='all', when='all'):
    '''Plot the fraction of choices of the highest probability for all three pairs presented during free choice
    trials'''
    
    sub_ave_fraction, mean_fraction_group, sem_fraction_group, _, _, _ = average_correct_choices(experiment, subject_IDs=subject_IDs, when=when)
    
    sub = list(zip(*sub_ave_fraction)) #for scatter plot, need to group data of each indiv for the same x (=couple of probas)

    #print(sub)
    
    #change resolution parameters for FENS 2022 Poster##  
    plt.rcParams['figure.dpi'] = 300
    plt.rcParams['savefig.dpi'] = 300  
    # change the default font family
    plt.rcParams.update({'font.family':'Arial'})
    
    width = 1 
    fig = plt.figure(1)
    ax = fig.add_subplot(111)
    ax = plt.gca()
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    #rects = ax.bar(0, var_75v25[0], width, yerr=var_75v25[1], color=[0,0,0,0], edgecolor='k', linewidth = 2)
    rects1 = ax.bar(1, mean_fraction_group[0], width, yerr=sem_fraction_group[0], color=[0.1,0.1,0.5,0.1], edgecolor='k', linewidth = 2)  ##if individual data points use (color=[0,0,0,0], edgecolor ='k', linewidth = 1) for the bar to be transparent
    rects2 = ax.bar(2, mean_fraction_group[1], width, yerr=sem_fraction_group[1], color=[0.1,0.1,0.5,0.1], edgecolor='k', linewidth = 2)
    rects3 = ax.bar(3, mean_fraction_group[2], width, yerr=sem_fraction_group[2], color=[0.1,0.1,0.5,0.1], edgecolor='k', linewidth = 2)
    plt.axhline(0.5, color='lightcoral')
    xTickMarks = ['100vs50', '100vs20', '50vs20']
    ax.set_xticks(range(1, 4))
    xtickNames = ax.set_xticklabels(xTickMarks, weight='bold')
    ax.tick_params(axis='x', which='major', labelsize=20)
    ax.tick_params(axis='y', which='major', labelsize=16)
    plt.ylim(0, 1)
    plt.ylabel('Fraction choice highest proba', fontsize=18)
    #plt.title('Cohort2')
    num_animals = len(sub_ave_fraction)
    colors = cm.rainbow(np.linspace(0, 1, num_animals)) #creates a palette of n_animals colors
    
    x_pos = range(1, len(mean_fraction_group)+1)
    x_for_scatter = [np.ones(num_animals) * x_pos[i-1] for i in x_pos] #multiply each x positions by the number of animals to have the good size for scatter plot
    
    for i in range(len(sub)):
        scatter = ax.scatter(x_for_scatter[i], sub[i], c=colors, s=4)
   
    #manually create legend for colors
    #all_patch = []
    #for i in range(len(colors)):
    #    all_patch.append(mpatches.Patch(color=colors[i], label=subject_IDs[i]))
    #plt.legend(handles=all_patch, loc='right', bbox_to_anchor=(1.2,0.5), fontsize=8)
    
    #use Line2D instead of patch to customize markers as circle in the legend
    all_lines = []
    for i in range(len(colors)):
        all_lines.append(Line2D(range(1), range(1), color='white', marker='o', markerfacecolor=colors[i])) #creates a line with a circle in middle, line put as white so it is not visible
    #plt.legend(all_lines,subject_IDs,numpoints=1, loc='right', bbox_to_anchor=(1.15,0.5), fontsize=7)
    
    for i in range(len(sub[0])):
        plt.plot([x_for_scatter[0][i], x_for_scatter[1][i], x_for_scatter[2][i]],[sub[0][i], sub[1][i], 
                 sub[2][i]],linewidth=0.1,color='k')     
        
        
    ###the following is used to plot the evolution of choices fraction across sessions (for each pair)
    ###
# =============================================================================
#     sessions_100v50 = [[] for i in range(len(fraction_individual))]
#     sessions_100v20 = [[] for i in range(len(fraction_individual))]
#     sessions_50v20 = [[] for i in range(len(fraction_individual))]
# 
#     for i in range(len(fraction_individual)):
#         for j in range(len(fraction_individual[i])):
#             sessions_100v50[i].append(fraction_individual[i][j][0])
#             sessions_100v20[i].append(fraction_individual[i][j][1])
#             sessions_50v20[i].append(fraction_individual[i][j][2])
#             
#     # get the maximum length of sessions
#     maxlen = len(max(sessions_100v50, key=len))
#     # pad left of each sublist with NaN to make it as long as the longest session
#     for sublist in sessions_100v50:
#         sublist[:] = sublist + [math.nan] * (maxlen - len(sublist))  
# 
#     # get the maximum length
#     maxlen = len(max(sessions_100v20, key=len))
#     # pad left of each sublist with NaN to make it as long as the longest session
#     for sublist in sessions_100v20:
#         sublist[:] = sublist + [math.nan] * (maxlen - len(sublist))
# 
#     # get the maximum length
#     maxlen = len(max(sessions_50v20, key=len))
#     # pad left of each sublist with NaN to make it as long as the longest session
#     for sublist in sessions_50v20:
#         sublist[:] = sublist + [math.nan] * (maxlen - len(sublist))
#             
#     #print('sessions100v50', sessions_100v50)
#     mean_100v50 = np.nanmean(sessions_100v50, axis=0)
#     mean_100v20 = np.nanmean(sessions_100v20, axis=0)
#     mean_50v20 = np.nanmean(sessions_50v20, axis=0)
#     
#     ######        
#         
#     #for graph of evolution of the choices across sessions
#     fig, (ax1, ax2, ax3) = plt.subplots(1, 3, sharey=True)
#     #fig = plt.figure(2)
#     x_ = range(1, len(sessions_100v50[0])+1)
#     #ax = fig.add_subplot(111)
#     #ax = plt.gca()
#     ax1.spines['top'].set_visible(False)
#     ax1.spines['right'].set_visible(False)
#     for i in range(len(sessions_100v50)):
#         #ax1.plot(x_, sessions_100v50[i], color=colors[i], marker='o', markersize=1)
#         ax1.plot(x_, sessions_100v50[i], color='silver', marker='o', markersize=1)
#     ax1.plot(x_, mean_100v50, color='blue', marker='o')
#     #ax1.axvline(x=3)    
#     ax1.set_xticks(range(1, len(sessions_100v50[0])+1))
#     ax1.set_title('100v50')
#     ax2.spines['top'].set_visible(False)
#     ax2.spines['right'].set_visible(False)
#     for i in range(len(sessions_100v20)):
#         #ax2.plot(x_, sessions_100v20[i], color=colors[i], marker='o') 
#         ax2.plot(x_, sessions_100v20[i], color='silver', marker='o', markersize=1)
#     ax2.plot(x_, mean_100v20, color='blue', marker='o')        
#     #ax2.axvline(x=3) 
#     ax2.set_xticks(range(1, len(sessions_100v20[0])+1))
#     ax2.set_title('100v20')    
#     ax3.spines['top'].set_visible(False)
#     ax3.spines['right'].set_visible(False)
#     for i in range(len(sessions_50v20)):
#         #ax3.plot(x_, sessions_50v20[i], color=colors[i], marker='o')
#         ax3.plot(x_, sessions_50v20[i], color='silver', marker='o', markersize=1)
#     ax3.plot(x_, mean_50v20, color='blue', marker='o')        
#     #ax3.axvline(x=3) 
#     ax3.set_xticks(range(1, len(sessions_50v20[0])+1)) 
#     ax3.set_title('50v20')    
#     plt.ylim(0, 1.1)
# =============================================================================
    
    #plt.ylabel('Fraction choice highest proba')

    #use Line2D instead of patch to customize markers as circle in the legend
    #all_lines = []
    #for i in range(len(colors)):
    #    all_lines.append(Line2D(range(1), range(1), color='white', marker='o', markerfacecolor=colors[i])) #creates a line with a circle in middle, line put as white so it is not visible
    #plt.legend(all_lines,subject_IDs,numpoints=1, loc='right', bbox_to_anchor=(1.5,0.5), fontsize=7)
    
##############################################################################################
        
        
        
def consecutive_events(session, event1_name, event2_name, all_events_names):
  # all_events_names = ['choose_left', 'choose_right', 'poke_1', 'poke_9']
  all_id, all_id_names = zip(*[(i, session.events[i].name) for i in range(len(session.events))
                               if session.events[i].name in all_events_names])

  event1_id = []
  event2_id = []
  pos = []
  pos1 = []
  for j in range(len(all_id_names)-1):
      if all_id_names[j] == event1_name and all_id_names[j+1] == event2_name:
          event1_id.append(all_id[j])
          event2_id.append(all_id[j+1])
          pos.append(j)
          pos1.append(j+1)      
  if event1_id == False:
      event1_id = np.nan
      event2_id = np.nan
      pos = np.nan
      pos1 = np.nan
  return event1_id, event2_id, pos, pos1

def get_times_consecutive_events(session, event1, event2, possible_events):
  '''
  event1: name of first event
  event2: name of the next consecutive event
  possible_events: list of names of possible events that could happen between these two events (including
                   event1 and event2), so you just get the time stamps when event1 and event2 happen
                   consecutively.
  return times of event1 and event2
  '''

  event1_id, event2_id, _, _ = consecutive_events(session, event1, event2, possible_events)

  times_e1 = [session.events[event1_id[i]][0] for i in range(len(event1_id))]
  times_e2 = [session.events[event2_id[i]][0] for i in range(len(event2_id))]
  return times_e1, times_e2


def get_times(session, id_list):
  times = [session.events[id_list[i]][0] for i in range(len(id_list))]
    
  return times

def NormalizeData(data):
    return (data - np.min(data)) / (np.max(data) - np.min(data))

def get_trial_side_details(center_out_to_side_ids, center_out_ids, choice_state_ids, stay_side_ids, init_id, ITI_id):
    center_out_side_ids = [x for i, x in enumerate(center_out_ids) if x in center_out_to_side_ids]
    choice_state_side_ids = [choice_state_ids[i] for i, x in enumerate(center_out_ids) if x in center_out_to_side_ids]
    poke_side_in_ids = [stay_side_ids[i] for i, x in enumerate(center_out_to_side_ids) if x in center_out_side_ids]
    
    real_choice_state_side_id = []
    real_side_in_id = []        
    trial_id_side = []
    for i, x in enumerate(init_id):
        temp_choice_state = [choice_state_side_ids[j] for j in range(len(choice_state_side_ids)) if init_id[i] < choice_state_side_ids[j] < ITI_id[i]]
        temp_side_in = [poke_side_in_ids[j] for j in range(len(choice_state_side_ids)) if init_id[i] < choice_state_side_ids[j] < ITI_id[i]]        
        if temp_choice_state:
            real_choice_state_side_id.append(temp_choice_state[0])
            trial_id_side.append(i)
        if temp_side_in:
            real_side_in_id.append(temp_side_in[0]) 
    
    trial_id_side = list(dict.fromkeys(trial_id_side)) #remove repeated values in list

    return center_out_side_ids, choice_state_side_ids, poke_side_in_ids, real_choice_state_side_id, real_side_in_id, trial_id_side


######################Latencies from initiation poke to reward pokes##########################
    
def latency_center_side(session, normalize=False, number='mean'):
    
    '''
    return the mean latencies travelling center poke to side poke
    parameters : 
    normalize: False (real time in ms) or True (take values normalized between 0 and 1)
    number= average of mean or median
    '''    
    print(session.subject_ID, session.datetime)
    
    free_choice_trials = session.trial_data['free_choice']  # 1 - free choice trial
                                                                     # 0 - forced choice trial       
    all_event_print = session.events_and_print
    all_prints = [all_event_print[i].name for i in range(len(all_event_print)) if type(all_event_print[i].name) == list and 'T' in all_event_print[i].name[0]] #take only print lines 
    trial_type = [all_prints[i][4].split(':')[1] for i in range(len(all_prints))] #all trial type, 'L', 'U, 'R' or 'FC'      
     
    #choice_state_id, conso_id, _, _ = consecutive_events(session, 'choice_state', 'reward_consumption', ['choice_state', 'reward_consumption'])
    init_id, ITI_id, _, _ = consecutive_events(session, 'init_trial', 'inter_trial_interval', ['init_trial', 'error_time_out', 'inter_trial_interval'])       
    choice_state_ids, poke_9_out_ids, _, _ = consecutive_events(session, 'choice_state', 'poke_9_out', ['choice_state', 'poke_9_out']) 
    poke_9_out_toL_ids, stay_left_ids, _, _ = consecutive_events(session, 'poke_9_out', 'stay_left', ['poke_9_out','stay_left', 'stay_up', 'stay_right'])
    poke_9_out_toU_ids, stay_up_ids, _, _ = consecutive_events(session, 'poke_9_out', 'stay_up', ['poke_9_out','stay_left', 'stay_up', 'stay_right'])    
    poke_9_out_toR_ids, stay_right_ids, _, _ = consecutive_events(session, 'poke_9_out', 'stay_right', ['poke_9_out','stay_left', 'stay_up', 'stay_right'])
    
    
    ##Choice_state as reference if we consider REACTION TIME after reward poke illuminates    
    center_out_L_ids, choice_state_L_ids, poke_left_ids, real_choice_state_L_id, real_left_in_id, trial_id_L = get_trial_side_details(poke_9_out_toL_ids, poke_9_out_ids, choice_state_ids, stay_left_ids, init_id, ITI_id)
    center_out_U_ids, choice_state_U_ids, poke_up_ids, real_choice_state_U_id, real_up_in_id, trial_id_U = get_trial_side_details(poke_9_out_toU_ids, poke_9_out_ids, choice_state_ids, stay_up_ids, init_id, ITI_id)
    center_out_R_ids, choice_state_R_ids, poke_right_ids, real_choice_state_R_id, real_right_in_id, trial_id_R = get_trial_side_details(poke_9_out_toR_ids, poke_9_out_ids, choice_state_ids, stay_right_ids, init_id, ITI_id)

#-------    
    choice_state_L_times = get_times(session, real_choice_state_L_id)
    choice_state_U_times = get_times(session, real_choice_state_U_id)
    choice_state_R_times = get_times(session, real_choice_state_R_id)
    left_in_times = get_times(session, real_left_in_id)
    up_in_times = get_times(session, real_up_in_id)
    right_in_times = get_times(session, real_right_in_id)
    
    latency_left = [left_in_times[i] - choice_state_L_times[i] for i in range(len(choice_state_L_times))]
    latency_up = [up_in_times[i] - choice_state_U_times[i] for i in range(len(choice_state_U_times))]
    latency_right = [right_in_times[i] - choice_state_R_times[i] for i in range(len(choice_state_R_times))]     
    #mean_lat_left = np.mean(latency_left)
    #mean_lat_up = np.mean(latency_up)
    #mean_lat_right = np.mean(latency_right)
    #sem_lat_left = stats.sem(latency_left)
    #sem_lat_up = stats.sem(latency_up)
    #sem_lat_right = stats.sem(latency_right)
    
    #Takes only lantencies for forced trials 
    lat_left_forced = [latency_left[i] for i, x in enumerate(trial_id_L) if free_choice_trials[x] == 0]  
    lat_up_forced = [latency_up[i] for i, x in enumerate(trial_id_U) if free_choice_trials[x] == 0]
    lat_right_forced = [latency_right[i] for i,x in enumerate(trial_id_R) if free_choice_trials[x] == 0]    

    #mean_lat_left_forced = np.mean(lat_left_forced)   #could be used to plot latencies towards the side rather than proba
    #mean_lat_up_forced = np.mean(lat_up_forced) 
    #mean_lat_right_forced = np.mean(lat_right_forced) 
    #sem_lat_left_forced = stats.sem(lat_left_forced)    
    #sem_lat_up_forced = stats.sem(lat_up_forced)
    #sem_lat_right_forced = stats.sem(lat_right_forced)    
        
    store_probas = session.store_probas
    probas = [float(store_probas[i]) for i in range(len(store_probas))]
    
    index_to_lat = [lat_left_forced, lat_up_forced, lat_right_forced] #maps the index of value in probas to the corresponding latency stored
    latency_forced = [index_to_lat[probas.index(val)] for val in [1, 0.5, 0.2]] #find the index of value in probas and use it to index in index_to_lat
    latency_100_forced, latency_50_forced, latency_20_forced = latency_forced

    mean_lat_100_forced = np.mean(latency_100_forced)#/1000 #divided by 1000 if want to have seconds and not ms
    mean_lat_50_forced = np.mean(latency_50_forced)
    mean_lat_20_forced = np.mean(latency_20_forced)    
    sem_lat_100_forced = stats.sem(latency_100_forced)    
    sem_lat_50_forced = stats.sem(latency_50_forced) 
    sem_lat_20_forced = stats.sem(latency_20_forced)

    diff_lat_10_50 = mean_lat_20_forced - mean_lat_50_forced 
    diff_lat_10_100 = mean_lat_20_forced - mean_lat_100_forced
    diff_lat_50_100 = mean_lat_50_forced - mean_lat_100_forced
    
    #normalize data
    #norm_lat_100_forced = NormalizeData(latency_100_forced)
    #norm_lat_50_forced = NormalizeData(latency_50_forced)
    #norm_lat_10_forced = NormalizeData(latency_10_forced)    
    
    #mean_norm_lat_100_forced = np.mean(norm_lat_100_forced)
    #mean_norm_lat_50_forced = np.mean(norm_lat_50_forced)
    #mean_norm_lat_10_forced = np.mean(norm_lat_10_forced)    
    #sem_norm_lat_100_forced = stats.sem(norm_lat_100_forced)   
    #sem_norm_lat_50_forced = stats.sem(norm_lat_50_forced) 
    #sem_norm_lat_10_forced = stats.sem(norm_lat_10_forced) 
    
    median_lat_100_forced = np.median(latency_100_forced) #median instead of mean latencies
    median_lat_50_forced = np.median(latency_50_forced)
    median_lat_20_forced = np.median(latency_20_forced)

    if normalize==False and number=='mean':
        return(mean_lat_100_forced, mean_lat_50_forced, mean_lat_20_forced, sem_lat_100_forced, sem_lat_50_forced, sem_lat_20_forced,
               diff_lat_10_50, diff_lat_10_100, diff_lat_50_100)
    elif normalize==True and number=='median':
        return(median_lat_100_forced, median_lat_50_forced, median_lat_20_forced)      

def average_latency_center_side(experiment, subject_IDs='all', when='all', normalize=False, number='mean', plot='indiv'):
    if subject_IDs == 'all': subject_IDs = sorted(experiment.subject_IDs)
    else:
        subject_IDs = sorted([experiment.subject_IDs[i] for i in range(len(experiment.subject_IDs)) if experiment.subject_IDs[i] in subject_IDs])
    #subject_IDs = experiment.subject_IDs
    subject_sessions = [experiment.get_sessions(subject_ID, when) for i, subject_ID in enumerate(subject_IDs)]
    
    lat_100_forced_indiv = [[latency_center_side(session, normalize=normalize, number=number)[0] for session in subject] for subject in subject_sessions]
    #print('lat 100', lat_100_forced_indiv)
    lat_50_forced_indiv = [[latency_center_side(session, normalize=normalize, number=number)[1] for session in subject] for subject in subject_sessions]
    lat_20_forced_indiv = [[latency_center_side(session, normalize=normalize, number=number)[2] for session in subject] for subject in subject_sessions]
    
    sessions = [lat_100_forced_indiv, lat_50_forced_indiv, lat_20_forced_indiv]
    for session_type in sessions: #loop through the different sessions to homogenise the length of sessions
        maxlen = len(max(session_type, key=len))
        for sublist in session_type:
            sublist[:] = sublist + [math.nan] * (maxlen - len(sublist))
    
    mean_100_by_day = np.nanmean(lat_100_forced_indiv, axis=0) #to show the evolution of latency across days
    mean_50_by_day = np.nanmean(lat_50_forced_indiv, axis=0)
    mean_20_by_day = np.nanmean(lat_20_forced_indiv, axis=0)    
    
    diff_lat_20_50_indiv = [[latency_center_side(session, normalize=normalize, number=number)[6] for session in subject] for subject in subject_sessions]
    diff_lat_20_100_indiv = [[latency_center_side(session, normalize=normalize, number=number)[7] for session in subject] for subject in subject_sessions]
    diff_lat_50_100_indiv = [[latency_center_side(session, normalize=normalize, number=number)[8] for session in subject] for subject in subject_sessions]
    
    mean_100_forced_for_indiv = [np.nanmean(lat_100_forced_indiv[i],axis = 0).tolist() for i, subject in enumerate(lat_100_forced_indiv)] # averaged across sessions for each subject
    mean_50_forced_for_indiv = [np.nanmean(lat_50_forced_indiv[i],axis = 0).tolist() for i, subject in enumerate(lat_50_forced_indiv)]
    mean_20_forced_for_indiv = [np.nanmean(lat_20_forced_indiv[i],axis = 0).tolist() for i, subject in enumerate(lat_20_forced_indiv)]
    
    mean_diff_20_50_for_indiv = [np.nanmean(diff_lat_20_50_indiv[i],axis = 0).tolist() for i, subject in enumerate(diff_lat_20_50_indiv)]
    mean_diff_20_100_for_indiv = [np.nanmean(diff_lat_20_100_indiv[i],axis = 0).tolist() for i, subject in enumerate(diff_lat_20_100_indiv)]
    mean_diff_50_100_for_indiv = [np.nanmean(diff_lat_50_100_indiv[i],axis = 0).tolist() for i, subject in enumerate(diff_lat_50_100_indiv)]
    sem_diff_20_50_for_indiv = [stats.sem(diff_lat_20_50_indiv[i],axis = 0).tolist() for i, subject in enumerate(diff_lat_20_50_indiv)]
    sem_diff_20_100_for_indiv = [stats.sem(diff_lat_20_100_indiv[i],axis = 0).tolist() for i, subject in enumerate(diff_lat_20_100_indiv)]
    sem_diff_50_100_for_indiv = [stats.sem(diff_lat_50_100_indiv[i],axis = 0).tolist() for i, subject in enumerate(diff_lat_50_100_indiv)]

    mean_100_forced_group = np.nanmean(mean_100_forced_for_indiv,axis=0).tolist() #average across all subjects
    mean_50_forced_group = np.nanmean(mean_50_forced_for_indiv,axis=0).tolist()
    mean_20_forced_group = np.nanmean(mean_20_forced_for_indiv,axis=0).tolist()       
    sem_100_forced_group = stats.sem(mean_100_forced_for_indiv,axis=0)
    sem_50_forced_group = stats.sem(mean_50_forced_for_indiv,axis=0)
    sem_20_forced_group = stats.sem(mean_20_forced_for_indiv,axis=0)     
    
    if plot=='indiv':
        return(mean_100_forced_for_indiv, mean_50_forced_for_indiv, mean_20_forced_for_indiv, mean_diff_50_100_for_indiv,
               mean_diff_20_100_for_indiv, mean_diff_20_50_for_indiv, sem_diff_50_100_for_indiv,
               sem_diff_20_100_for_indiv, sem_diff_20_50_for_indiv, lat_100_forced_indiv, lat_50_forced_indiv, lat_20_forced_indiv, 
               mean_100_by_day, mean_50_by_day, mean_20_by_day, subject_IDs)
    elif plot == 'group':
        return(mean_100_forced_group, mean_50_forced_group, mean_20_forced_group, sem_100_forced_group, sem_50_forced_group, sem_20_forced_group)


def plot_latencies_center_side(experiment, subject_IDs='all', when='all', normalize=False, number='mean', form='lat'):
    '''plot the latencies to reach a reward poke after initiating the trial based on the probability associated
    with the reward poke'''
    
    latencies_group = average_latency_center_side(experiment, subject_IDs=subject_IDs, when=when, normalize=False, number=number, plot='group')
    latencies_indiv = average_latency_center_side(experiment, subject_IDs=subject_IDs, when=when, normalize=False, number=number, plot='indiv')   

    #change resolution parameters for FENS Poster##  
    plt.rcParams['figure.dpi'] = 300
    plt.rcParams['savefig.dpi'] = 300  
    # change the default font family
    plt.rcParams.update({'font.family':'Arial'})    
    
    if form == 'lat':
        fig = plt.figure()
        ax = fig.add_subplot(111)
        ax = plt.gca()
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)
        rects = ax.bar(0, latencies_group[0], yerr=latencies_group[3], color=[0.1,0.1,0.5,0.1], edgecolor='k', linewidth = 2)
        rects1 = ax.bar(1, latencies_group[1], yerr=latencies_group[4], color=[0.1,0.1,0.5,0.1], edgecolor='k', linewidth = 2)  ##if individual data points use (color=[0,0,0,0], edgecolor ='k', linewidth = 1) for the bar to be transparent
        rects2 = ax.bar(2, latencies_group[2], yerr=latencies_group[5], color=[0.1,0.1,0.5,0.1], edgecolor='k', linewidth = 2)
        xTickMarks = ['100%','50%','20%']
        ax.set_xticks(range(3))
        xtickNames = ax.set_xticklabels(xTickMarks, weight = 'bold')
        ax.tick_params(axis='x', which='major', labelsize=20)
        ax.tick_params(axis='y', which='major', labelsize=16)        
        plt.ylabel('Latency center to side (ms)', fontsize=18)
        #plt.title(title)
        #plt.ylim([0, 4000])
        num_animals = len(latencies_indiv[0])
        colors = cm.rainbow(np.linspace(0, 1, num_animals))
        
        pos1 = [np.ones(num_animals)*0] ##for each x positions, multiply by the number of animals to have the right number of points on the same x pos
        pos2 = [np.ones(num_animals)*1]
        pos3 = [np.ones(num_animals)*2]
        
        ax.scatter(pos1, latencies_indiv[0], c=colors, s=4)
        ax.scatter(pos2, latencies_indiv[1], c=colors, s=4)
        ax.scatter(pos3, latencies_indiv[2], c=colors, s=4)
        
        #use Line2D instead of patch to customize markers as circle in the legend
        all_lines = []
        for i in range(len(colors)):
            all_lines.append(Line2D(range(1), range(1), color='white', marker='o', markerfacecolor=colors[i])) #creates a line with a circle in middle, line put as white so it is not visible
        #plt.legend(all_lines,latencies_indiv[-1],numpoints=1, loc='right', bbox_to_anchor=(1.15,0.5), fontsize=7)        

        for i in range(len(latencies_indiv[0])):
            plt.plot([pos1[0][i], pos2[0][i], pos3[0][i]],[latencies_indiv[0][i], latencies_indiv[1][i], 
                     latencies_indiv[2][i]],linewidth=0.1,color='k')    
            
    elif form == 'diff':
        fig = plt.figure()
        ax = fig.add_subplot(111)
        ax = plt.gca()
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)
        x_pos = np.arange(len(latencies_indiv[0]))
        rects = ax.bar(x_pos, latencies_indiv[3], yerr=latencies_indiv[6], color=[0,0,0,0], edgecolor='k', linewidth = 2)
        #rects1 = ax.bar(x_pos, latencies_indiv[4], yerr=latencies_indiv[7], color=[0,0,0,0], edgecolor='k', linewidth = 2)  ##if individual data points use (color=[0,0,0,0], edgecolor ='k', linewidth = 1) for the bar to be transparent
        #rects2 = ax.bar(x_pos, latencies_indiv[5], yerr=latencies_indiv[8], color=[0,0,0,0], edgecolor='k', linewidth = 2)  
        xTickMarks = latencies_indiv[-1]
        ax.set_xticks(range(len(xTickMarks)))
        #ax.tick_params(axis='both', which='major', labelsize=8)
        xtickNames = ax.set_xticklabels(xTickMarks, fontsize = 8, rotation='vertical')
        plt.ylabel('Diff Latencies (ms)')
        plt.title('Diff 50% - 100%')

    elif form == 'evolution':
        #for graph of evolution of the choices across sessions
        fig, (ax1, ax2, ax3) = plt.subplots(1, 3, sharey=True)
        #fig = plt.figure(2)
        x_ = range(1, len(latencies_indiv[9][0])+1)
        #print(x_)
        #ax = fig.add_subplot(111)
        #ax = plt.gca()
        ax1.spines['top'].set_visible(False)
        ax1.spines['right'].set_visible(False)
        for i in range(len(latencies_indiv[9])):
            #ax1.plot(x_, sessions_100v50[i], color=colors[i], marker='o', markersize=1)
            ax1.plot(x_, latencies_indiv[9][i], color='silver', marker='o', markersize=1)
        ax1.plot(x_, latencies_indiv[12], color='blue', marker='o')
        #ax1.axvline(x=3)    
        ax1.set_xticks(range(1, len(latencies_indiv[9][0])+1))
        ax1.set_title('Latency 100%')
        
        ax2.spines['top'].set_visible(False)
        ax2.spines['right'].set_visible(False)
        for i in range(len(latencies_indiv[10])):
            #ax2.plot(x_, sessions_100v20[i], color=colors[i], marker='o') 
            ax2.plot(x_, latencies_indiv[10][i], color='silver', marker='o', markersize=1)
        ax2.plot(x_, latencies_indiv[13], color='blue', marker='o')        
        #ax2.axvline(x=3) 
        ax2.set_xticks(range(1, len(latencies_indiv[10][0])+1))
        ax2.set_title('Latency 50%')   
        
        ax3.spines['top'].set_visible(False)
        ax3.spines['right'].set_visible(False)
        for i in range(len(latencies_indiv[11])):
            #ax3.plot(x_, sessions_50v20[i], color=colors[i], marker='o')
            ax3.plot(x_, latencies_indiv[11][i], color='silver', marker='o', markersize=1)
        ax3.plot(x_, latencies_indiv[14], color='blue', marker='o')        
        #ax3.axvline(x=3) 
        ax3.set_xticks(range(1, len(latencies_indiv[11][0])+1)) 
        ax3.set_title('latency 20%')    
        #plt.ylim(0, 4500)            
        
        
######################Breaks in the reward pokes##########################
###########################################################################

        
def proportion_trials_with_breaks(session):
    '''Code extract all break events. If at least one break in a trial, 1 is assigned it to trial types 100%, 50% or 20%,
    percentage of trials with at least one break is then calculated based on this'''
    
    print(session.subject_ID, session.datetime_string)
    all_event_print = session.events_and_print
    all_prints_trial = [all_event_print[i].name for i in range(len(all_event_print)) if type(all_event_print[i].name) == list and 'T' in all_event_print[i].name[0]] #take only print lines and exclude the one announcing n of trials before reversal
    trial_type = [all_prints_trial[i][4].split(':')[1] for i in range(len(all_prints_trial))] #all trial type     
    all_prints_holdingtimes = [all_event_print[i].name for i in range(len(all_event_print)) if type(all_event_print[i].name) == list and 'T' not in all_event_print[i].name[0]]
    proba_choosed = session.trial_data['proba_choosed']
    
    init_id, ITI_id, _, _ = consecutive_events(session, 'init_trial', 'inter_trial_interval', ['init_trial', 'error_time_out', 'inter_trial_interval'])    
    
    #all_grace_L_id = [i for i in range(len(session.events)) if session.events[i].name == 'grace_left']
    #all_grace_U_id = [i for i in range(len(session.events)) if session.events[i].name == 'grace_up']
    #all_grace_R_id = [i for i in range(len(session.events)) if session.events[i].name == 'grace_right']
    
    all_grace_id = [i for i in range(len(session.events)) if session.events[i].name == 'grace_left' or session.events[i].name == 'grace_up' or session.events[i].name == 'grace_right']
    
    breaks_by_trial = [[j for j in all_grace_id if init_id[i] < j < ITI_id[i]] for i in range(len(init_id))]
    
    trials_100_with_breaks = [1 if breaks_by_trial[i] else 0 for i, x in enumerate(breaks_by_trial) if proba_choosed[i] == 1] #1 if at least one break in the trial, 0 if not
    trials_50_with_breaks = [1 if breaks_by_trial[i] else 0 for i, x in enumerate(breaks_by_trial) if proba_choosed[i] == 0.5]
    trials_20_with_breaks = [1 if breaks_by_trial[i] else 0 for i, x in enumerate(breaks_by_trial) if proba_choosed[i] == 0.2] 
 
    trials = [trials_100_with_breaks, trials_50_with_breaks, trials_20_with_breaks]
    proportions = []
    for trial in trials:
        if 1 in trial:
            proportion = (sum(trial) / len(trial)) * 100 #percentage of trials with at least one break
        else:
            proportion = 0
        proportions.append(proportion)
        
    proportion_100_breaks, proportion_50_breaks, proportion_20_breaks = proportions

       
    return proportion_100_breaks, proportion_50_breaks, proportion_20_breaks


def average_prop_side_breaks(experiment, subject_IDs='all', when='all', plot='indiv'):
    if subject_IDs == 'all': subject_IDs = sorted(experiment.subject_IDs)
    else:
        subject_IDs = sorted([experiment.subject_IDs[i] for i in range(len(experiment.subject_IDs)) if experiment.subject_IDs[i] in subject_IDs])
    #subject_IDs = experiment.subject_IDs
    subject_sessions = [experiment.get_sessions(subject_ID, when) for i, subject_ID in enumerate(subject_IDs)]

    break_prop_100_indiv = [[proportion_trials_with_breaks(session)[0] for session in subject] for subject in subject_sessions]
    break_prop_50_indiv = [[proportion_trials_with_breaks(session)[1] for session in subject] for subject in subject_sessions]
    break_prop_20_indiv = [[proportion_trials_with_breaks(session)[2] for session in subject] for subject in subject_sessions]

    mean_prop_100_for_indiv = [np.nanmean(break_prop_100_indiv[i],axis = 0).tolist() for i, subject in enumerate(break_prop_100_indiv)] # averaged across sessions for each subject
    mean_prop_50_for_indiv = [np.nanmean(break_prop_50_indiv[i],axis = 0).tolist() for i, subject in enumerate(break_prop_50_indiv)]
    mean_prop_20_for_indiv = [np.nanmean(break_prop_20_indiv[i],axis = 0).tolist() for i, subject in enumerate(break_prop_20_indiv)]
 
    mean_prop_100_group = np.nanmean(mean_prop_100_for_indiv,axis=0).tolist() #average across all subjects
    mean_prop_50_group = np.nanmean(mean_prop_50_for_indiv,axis=0).tolist()
    mean_prop_20_group = np.nanmean(mean_prop_20_for_indiv,axis=0).tolist()       
    sem_prop_100_group = stats.sem(mean_prop_100_for_indiv,axis=0)
    sem_prop_50_group = stats.sem(mean_prop_50_for_indiv,axis=0) 
    sem_prop_20_group = stats.sem(mean_prop_20_for_indiv,axis=0)

    if plot=='indiv':
        return(mean_prop_100_for_indiv, mean_prop_50_for_indiv, mean_prop_20_for_indiv, subject_IDs)
    elif plot == 'group':
        return(mean_prop_100_group, mean_prop_50_group, mean_prop_20_group, sem_prop_100_group,
               sem_prop_50_group, sem_prop_20_group)


def plot_side_breaks(experiment, subject_IDs = 'all', when='all'):
    '''Plot the proportion of trial with a break in reward poke for each probability trial type (100%, 50%, 20%)'''
    fail_group = average_prop_side_breaks(experiment, subject_IDs='all', when='all',plot='group')
    fail_indiv = average_prop_side_breaks(experiment, subject_IDs='all', when='all',plot='indiv')
    
    #change resolution parameters for FENS Poster##  
    plt.rcParams['figure.dpi'] = 300
    plt.rcParams['savefig.dpi'] = 300  
    # change the default font family
    plt.rcParams.update({'font.family':'Arial'})
    
    fig = plt.figure(1)
    ax = fig.add_subplot(111)
    ax = plt.gca()
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    rects = ax.bar(0, fail_group[0], yerr=fail_group[3], color=[0.1,0.1,0.5,0.1], edgecolor='k', linewidth = 2)
    rects1 = ax.bar(1, fail_group[1], yerr=fail_group[4], color=[0.1,0.1,0.5,0.1], edgecolor='k', linewidth = 2)  ##if individual data points use (color=[0,0,0,0], edgecolor ='k', linewidth = 1) for the bar to be transparent
    rects2 = ax.bar(2, fail_group[2], yerr=fail_group[5], color=[0.1,0.1,0.5,0.1], edgecolor='k', linewidth = 2)
    xTickMarks = ['100%', '50%','20%']
    ax.set_xticks(range(3))
    xtickNames = ax.set_xticklabels(xTickMarks, weight='bold')
    
    ax.tick_params(axis='x', which='major', labelsize=20)
    ax.tick_params(axis='y', which='major', labelsize=16)
    plt.ylabel('% of trials with at least one break', fontsize = 18)
    plt.ylim([0, 100])
    #plt.title(title)
    num_animals = len(fail_indiv[0])
    colors = cm.rainbow(np.linspace(0, 1, num_animals))
    
    pos1 = [np.ones(num_animals)*0] ##for each x positions, multiply by the number of animals to have the right number of points on the same x pos
    pos2 = [np.ones(num_animals)*1]
    pos3 = [np.ones(num_animals)*2]
    
    ax.scatter(pos1, fail_indiv[0], c=colors, s=4)
    ax.scatter(pos2, fail_indiv[1], c=colors, s=4)
    ax.scatter(pos3, fail_indiv[2], c=colors, s=4)   
    
    #use Line2D instead of patch to customize markers as circle in the legend
    all_lines = []
    for i in range(len(colors)):
        all_lines.append(Line2D(range(1), range(1), color='white', marker='o', markerfacecolor=colors[i])) #creates a line with a circle in middle, line put as white so it is not visible
    #plt.legend(all_lines,fail_indiv[-1],numpoints=1, loc='right', bbox_to_anchor=(1.15,0.5), fontsize=7)
    
    for i in range(len(fail_indiv[0])):
        plt.plot([pos1[0][i], pos2[0][i], pos3[0][i]],[fail_indiv[0][i], fail_indiv[1][i], 
                 fail_indiv[2][i]],linewidth=0.1,color='k')         
        
        
##################################################################################        
##################################################################################
#Correlation between forced choices latencies and free choices

def ScaleData(data):
    return ((2 *(data - np.min(data))/(np.max(data)-np.min(data))) - 1)

def plot_correlation_latency_choice(experiment, subject_IDs='all', when='all'):#(choices, difflat_100v50, difflat_100v20, difflat_50v20):
    '''Function runs a Pearson correlation test between data points corresponding to
    free choices and forced trials latencies (differences) for each individual and plot the results'''
    
    choices = average_correct_choices(experiment, subject_IDs=subject_IDs, when=when)[0]
    diff_lat_100v50 = average_latency_center_side(experiment, subject_IDs=subject_IDs, when=when)[3]
    diff_lat_100v20 = average_latency_center_side(experiment, subject_IDs=subject_IDs, when=when)[4]
    diff_lat_50v20 = average_latency_center_side(experiment, subject_IDs=subject_IDs, when=when)[5]
    
    scale_lat_100v50 = ScaleData(diff_lat_100v50)
    scale_lat_100v20 = ScaleData(diff_lat_100v20)
    scale_lat_50v20 = ScaleData(diff_lat_50v20)
    
    choice_100v50 = [choices[i][0] for i in range(len(choices))]
    choice_100v20 = [choices[i][1] for i in range(len(choices))]
    choice_50v20 = [choices[i][2] for i in range(len(choices))]
    
    #create dataframe with choices and latencies difference
    df = pd.DataFrame({'choice_100vs50':choice_100v50, 'latency_100vs50': scale_lat_100v50,
                       'choice_100vs20':choice_100v20, 'latency_100vs20':scale_lat_100v20,
                       'choice_50vs20':choice_50v20, 'latency_50vs20':scale_lat_50v20})

    plt.figure(1)
    result_corr_100v50 = stats.pearsonr(df['choice_100vs50'], df['latency_100vs50'])
    sns.lmplot(x="choice_100vs50", y="latency_100vs50", data=df, scatter_kws={"color": "black"}, line_kws={"color": "blue"})
    round_corr_100v50 = round(result_corr_100v50[0], 3)
    round_p_100v50 = round(result_corr_100v50[1], 3)
    plt.figtext(0.2, 0.83, f'r = {round_corr_100v50}')
    plt.figtext(0.2, 0.78, f'p = {round_p_100v50}')
    plt.title('100vs50')
    
    plt.figure(2)
    result_corr_100v20 = stats.pearsonr(df['choice_100vs20'], df['latency_100vs20'])
    sns.lmplot(x="choice_100vs20", y="latency_100vs20", data=df, scatter_kws={"color": "black"}, line_kws={"color": "red"})
    round_corr_100v20 = round(result_corr_100v20[0], 3)
    round_p_100v20 = round(result_corr_100v20[1], 3)
    plt.figtext(0.2, 0.83, f'r = {round_corr_100v20}')
    plt.figtext(0.2, 0.78, f'p = {round_p_100v20}')
    plt.title('100vs20')
    
    plt.figure(3)
    result_corr_50v20 = stats.pearsonr(df['choice_50vs20'], df['latency_50vs20'])
    sns.lmplot(x="choice_50vs20", y="latency_50vs20", data=df, scatter_kws={"color": "black"}, line_kws={"color": "green"})
    round_corr_50v20 = round(result_corr_50v20[0], 3)
    round_p_50v20 = round(result_corr_50v20[1], 3)
    plt.figtext(0.2, 0.83, f'r = {round_corr_50v20}')
    plt.figtext(0.2, 0.78, f'p = {round_p_50v20}')
    plt.title('50vs20')