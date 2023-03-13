# -*- coding: utf-8 -*-
"""
Created on Fri Feb 10 15:22:29 2023

@author: Cerpa Juan Carlos
"""

import sys, os
import numpy as np
import pandas as pd
import math
import warnings
import statsmodels
from scipy import interpolate
from scipy import stats
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from sklearn import linear_model as lm

def select_trial_types_to_analyse(session, region='DMS', hemisphere='L'):
  '''This code go through all session trials and assign every trial to different trial types
  which can be then used to plot photometry signals corresponding to these trial types.
  Example of trial types: trial rewarded or non rewarded, trial where the reward poke choosed
  is contralateral or ipsilateral to the recording site, trial where the reward poke choosed is
  associated to 100%, 50% or 20% probability'''
  
  subject_ID = session.subject_ID  
  all_event_print = session.events_and_print
  all_prints = [all_event_print[i].name for i in range(len(all_event_print)) if type(all_event_print[i].name) == list and 'T' in all_event_print[i].name[0]] #take only all the print lines appearing at the end of a trial
  trial_type = [all_prints[i][4].split(':')[1] for i in range(len(all_prints))] #all trial type, 'FC'
  choices = session.trial_data['choices']  # 1 - left; 2 - up; 3 - right
  outcomes = session.trial_data['outcomes']  # 1 - rewarded;  0 - non-rewarded
  free_choice_trials = session.trial_data['free_choice']  # 1 - free choice trial
  proba_choosed = session.trial_data['proba_choosed'] #all proba choosed
  
  hemisphere_param = [1 if hemisphere == 'L' else -1][0]
  ipsi_contra_choice = np.asarray([0.5 * hemisphere_param if c == 1 else -0.5 * hemisphere_param if c == 3 else 0 for c in (choices * 1)])
  
  ipsi_choice = np.where(ipsi_contra_choice == 0.5)[0] #ipsilateral trials
  contra_choice = np.where(ipsi_contra_choice == -0.5)[0]
  up_choice = np.where(ipsi_contra_choice == 0)[0]    
                                                                
  free_trial = np.where(free_choice_trials == 1)[0]
  forced_trial = np.where(free_choice_trials == 0)[0]
  
  free_choices = np.array([choices[x] for x in range(len(choices)) if x in free_trial]) #all choices during free trials
  forced_choices = np.array([choices[x] for x in range(len(choices)) if x in forced_trial]) #all choices during forced trials
  
  left_choice = np.where(choices == 1)[0] #when choice was left
  up_choice = np.where(choices == 2)[0]
  right_choice = np.where(choices == 3)[0]
  
  reward_trials = np.where(outcomes == 1)[0] #all rewarded trials
  nonreward_trials = np.where(outcomes == 0)[0]  
  
  trial_100 = [i for i, x in enumerate(proba_choosed) if x == 1] #trials where 100% is presented and selected, both forced and free trials
  trial_50 = [i for i, x in enumerate(proba_choosed) if x == 0.5]
  trial_20 = [i for i, x in enumerate(proba_choosed) if x == 0.2]  

  contra_100 = np.array([x for x in contra_choice if x in trial_100]) #100% trials when in contralateral side
  contra_50 = np.array([x for x in contra_choice if x in trial_50])
  contra_20 = np.array([x for x in contra_choice if x in trial_20])
  
  contra_100_forced = np.array([x for x in contra_100 if x in forced_trial]) #100% forced trials when in contralateral side
  contra_50_forced = np.array([x for x in contra_50 if x in forced_trial])
  contra_20_forced = np.array([x for x in contra_20 if x in forced_trial])

  contra_100_free = np.array([x for x in contra_100 if x in free_trial]) #100% free trials when in contralateral side
  contra_50_free = np.array([x for x in contra_50 if x in free_trial])
  contra_20_free = np.array([x for x in contra_20 if x in free_trial])
  
  ipsi_100 = np.array([x for x in ipsi_choice if x in trial_100])
  ipsi_50 = np.array([x for x in ipsi_choice if x in trial_50])
  ipsi_20 = np.array([x for x in ipsi_choice if x in trial_20])
  
  ipsi_100_forced = np.array([x for x in ipsi_100 if x in forced_trial])
  ipsi_50_forced = np.array([x for x in ipsi_50 if x in forced_trial])
  ipsi_20_forced = np.array([x for x in ipsi_20 if x in forced_trial])

  ipsi_100_free = np.array([x for x in ipsi_100 if x in free_trial])
  ipsi_50_free = np.array([x for x in ipsi_50 if x in free_trial])
  ipsi_20_free = np.array([x for x in ipsi_20 if x in free_trial])
  
  up_100 = np.array([x for x in up_choice if x in trial_100])
  up_50 = np.array([x for x in up_choice if x in trial_50])
  up_20 = np.array([x for x in up_choice if x in trial_20])  
  
  up_100_forced = np.array([x for x in up_100 if x in forced_trial])
  up_50_forced = np.array([x for x in up_50 if x in forced_trial])
  up_20_forced = np.array([x for x in up_20 if x in forced_trial])

  up_100_free = np.array([x for x in up_100 if x in free_trial])
  up_50_free = np.array([x for x in up_50 if x in free_trial])
  up_20_free = np.array([x for x in up_20 if x in free_trial])
  
  high_prob = []
  low_prob = []
  for i,x in enumerate(trial_type):
      if x == 'FC':
          high_prob.append(float(all_prints[i][6].split(':')[1])) #if free choice trials, high prob is a float
          low_prob.append(float(all_prints[i][7].split(':')[1]))
      else:
          high_prob.append(str(all_prints[i][6].split(':')[1])) #if forced trials, prob is a string (None)
          low_prob.append(str(all_prints[i][7].split(':')[1]))

  #ids of free trials where the chosen option at trial t-1 is offered again at trial t
  poke_choosed_repeat_ids = []
  poke_choosed_norepeat_ids = []
  for i in range(1, len(proba_choosed)):
      if trial_type[i-1] == 'FC' and trial_type[i] == 'FC':
          if proba_choosed[i-1] == high_prob[i] or proba_choosed[i-1] == low_prob[i]:
              poke_choosed_repeat_ids.append(i)
          elif proba_choosed[i-1] != high_prob[i] and proba_choosed[i-1] != low_prob[i]:
              poke_choosed_norepeat_ids.append(i)
              
              
  #ids of free trial where the animal choose the same option than trial t-1
  same_free_choice = []
  diff_free_choice = []
  for i in range(1, len(proba_choosed)):
      if trial_type[i-1] == 'FC' and trial_type[i] == 'FC':
          if proba_choosed[i] == proba_choosed[i-1]:
              same_free_choice.append(i)            
          elif proba_choosed[i] != proba_choosed[i-1]:
              diff_free_choice.append(i)

  #ids of free trials where the animal do not choose the poke that is presented again
  repeat_notchoosed_poke = []
  for i in range(len(choices)):
      if i in poke_choosed_repeat_ids:
          if i in diff_free_choice:
              repeat_notchoosed_poke.append(i)

  repeat_forced_trial_ids = [] #store forced trial ids where the single reward poke illuminated is the same as in the previous forced trial
  different_forced_trial_ids = []
  for i in range(1, len(trial_type)):
      if trial_type[i] in ['L', 'U', 'R']:
          if trial_type[i] == trial_type[i-1]:
              repeat_forced_trial_ids.append(i)
          else:
              different_forced_trial_ids.append(i)
  
  repeat_forced_50 = [i for i in repeat_forced_trial_ids if proba_choosed[i-1] == 1]
  repeat_forced_100 = [i for i in repeat_forced_trial_ids if proba_choosed[i-1] == 0.5]
  repeat_forced_20 = [i for i in repeat_forced_trial_ids if proba_choosed[i-1] == 0.2]

  ids_100v50 = [i for i in range(len(high_prob)) if high_prob[i] == 1.0 and low_prob[i] == 0.5] #ids of (free) trials where the pair presented was 100vs50
  ids_100v10 = [i for i in range(len(high_prob)) if high_prob[i] == 1.0 and low_prob[i] == 0.2]
  ids_50v10 = [i for i in range(len(high_prob)) if high_prob[i] == 0.5 and low_prob[i] == 0.2]
    
  correct_choices = [1 if x == high_prob[i] else 0 for i, x in enumerate(proba_choosed)]    
      
  correct_trials = [i for i in range(len(correct_choices)) if correct_choices[i] == 1]     

  correct_100v50 = np.array([x for x in ids_100v50 if x in correct_trials])
  uncorrect_100v50 = np.array([x for x in ids_100v50 if x not in correct_trials])
 
  trial_100_forced = np.array([x for x in trial_100 if x in forced_trial])
  trial_50_forced = np.array([x for x in trial_50 if x in forced_trial])
  trial_20_forced = np.array([x for x in trial_20 if x in forced_trial])
  
  trial_100_free = np.array([x for x in trial_100 if x in free_trial])
  trial_50_free = np.array([x for x in trial_50 if x in free_trial])
  trial_20_free = np.array([x for x in trial_20 if x in free_trial])

  trial_50_rew = np.array([x for x in trial_50 if x in reward_trials]) #all forced and free 50% trials, rewarded
  trial_50_nonrew = np.array([x for x in trial_50 if x in nonreward_trials]) #all forced and free 50% trials, non rewarded
  trial_20_rew = np.array([x for x in trial_20 if x in reward_trials])
  trial_20_nonrew = np.array([x for x in trial_20 if x in nonreward_trials])

  forced_50_rew = np.array([x for x in trial_50_forced if x in reward_trials]) #only forced 50% trials, rewarded
  forced_50_nonrew = np.array([x for x in trial_50_forced if x in nonreward_trials]) #only forced 50% trials, non rewarded
  forced_20_rew = np.array([x for x in trial_20_forced if x in reward_trials])
  forced_20_nonrew = np.array([x for x in trial_20_forced if x in nonreward_trials])  

  chunked_free_trial = [free_trial[i:i+10] for i in range(0,len(free_trial), 10)] #format free trials as blocks of free trials occurring during session
  free_outcome = np.array([outcomes[x] for x in range(len(outcomes)) if x in free_trial]) #all outcome during free trials, 1 is reward, 0 is no reward
  chunked_free_outcome = [free_outcome[i:i+10] for i in range(0,len(free_outcome), 10)] #split the miniblocks of free outcomes in sublists
  chunked_free_choice = [free_choices[i:i+10] for i in range(0,len(free_choices), 10)] #split the miniblocks of free choice in sublists          

  return {'all': [x for x in range(len(choices))],
          'free_trial': free_trial,
          'forced_trial': forced_trial,
          'trial_100': trial_100,
          'trial_50': trial_50,
          'trial_20': trial_20,
          'trial_100_forced': trial_100_forced,
          'trial_50_forced': trial_50_forced,
          'trial_20_forced': trial_20_forced,
          'trial_100_free': trial_100_free,
          'trial_50_free': trial_50_free,
          'trial_20_free': trial_20_free,            
          'ipsi_100': ipsi_100,
          'ipsi_50': ipsi_50,
          'ipsi_20': ipsi_20,
          'ipsi_100_forced': ipsi_100_forced,
          'ipsi_50_forced': ipsi_50_forced,
          'ipsi_20_forced': ipsi_20_forced,   
          'ipsi_100_free': ipsi_100_free,
          'ipsi_50_free': ipsi_50_free,
          'ipsi_20_free': ipsi_20_free,          
          'contra_100': contra_100,
          'contra_50': contra_50,
          'contra_20': contra_20,
          'contra_100_forced': contra_100_forced,
          'contra_50_forced': contra_50_forced,
          'contra_20_forced': contra_20_forced,
          'contra_100_free': contra_100_free,
          'contra_50_free': contra_50_free,
          'contra_20_free': contra_20_free,          
          'up_100': up_100,
          'up_50': up_50,
          'up_20': up_20,
          'up_100_forced': up_100_forced,
          'up_50_forced': up_50_forced,
          'up_20_forced': up_20_forced, 
          'up_100_free': up_100_free,
          'up_50_free': up_50_free,
          'up_20_free': up_20_free,          
          'trial_100v50': ids_100v50,
          'correct_100v50': correct_100v50,
          'uncorrect_100v50': uncorrect_100v50,
          'ipsilateral trial': ipsi_choice,
          'contralateral trial': contra_choice,
          'up trial': up_choice, 
          'same_free_choice': same_free_choice,
          'diff_free_choice': diff_free_choice,
          'reward_trials': reward_trials,
          'nonreward_trials': nonreward_trials,
          'trial_50_rew': trial_50_rew,
          'trial_20_rew': trial_20_rew,
          'trial_50_nonrew': trial_50_nonrew,
          'trial_20_nonrew': trial_20_nonrew,
          'forced_50_rew': forced_50_rew,
          'forced_50_nonrew': forced_50_nonrew,
          'forced_20_rew': forced_20_rew,
          'forced_20_nonrew': forced_20_nonrew,
          'choosed_poke_reproposed': poke_choosed_repeat_ids,
          'choosed_poke_notproposed': poke_choosed_norepeat_ids,
          'reproposed_poke_notselected': repeat_notchoosed_poke,
          'repeat_forced_trials': repeat_forced_trial_ids,
          'different_forced_trials': different_forced_trial_ids}

def get_id_trials_to_analyse_all_sessions(sessions, trial_type, region, hemisphere):
  trial_id_sessions = []
  for i, session in enumerate(sessions):
    dict_id = select_trial_types_to_analyse(session, region[i], hemisphere[i])
    trial_id_sessions.append(dict_id[trial_type])

  return trial_id_sessions

def get_dict_event_times(session, dict_events_id):
  # get times when events happened
  dict_events_times = {}
  for x in dict_events_id:
    dict_events_times[x] = [session.events[dict_events_id[x][i]][0] for i in range(len(dict_events_id[x]))]
  return dict_events_times 

def get_index_time(photometry, time):
  '''
  photometry: time of photometry signal
  time: time stamp to find in the photometry signal
  return index where time is in photometry signal time
  '''
  idx = np.nanargmin(np.abs(photometry - time))

  # print(np.nanargmin(np.abs(photometry - time))) ########################################################################

  return idx

def consecutive_events(session, event1_name, event2_name, all_events_names):
  '''Function extract all specified events of the session (all_events_names) and look only for consecutive specified
  events (event1_name and event2_name)'''
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

def scale_time_events(median_time, median_len, corrected_signal, pho_id1_i, pho_id2_i):
  '''
  take as parameters, median time length and median length of time points and scale all trials to a common time
  '''

  if (pho_id1_i == pho_id2_i) or (pho_id1_i + 1 == pho_id2_i) or (corrected_signal[pho_id1_i:pho_id2_i].size == 0):
    # fill with the same value
    x = np.linspace(0, median_time, median_len)
    y_stretch = [corrected_signal[pho_id1_i]] * len(x)
    # return [], []
  else:
    x = np.linspace(0, median_time, median_len)
    y_interp = interpolate.interp1d(np.arange(corrected_signal[pho_id1_i:pho_id2_i].size),
                                    corrected_signal[pho_id1_i:pho_id2_i], fill_value='extrapolate')
    y_stretch = y_interp(np.linspace(0, corrected_signal[pho_id1_i:pho_id2_i].size - 1, x.size))
  return x, y_stretch

def get_index_photometry_events(session, sample_times_pyc, dict_events_times, trial_time_point_align):
  '''return dict with indexes in the photometry signal when events happened'''
  dict_pho_events_id = {}
  for x in trial_time_point_align:
    dict_pho_events_id[x] = [get_index_time(
      sample_times_pyc, dict_events_times[trial_time_point_align[x][0]][i] + trial_time_point_align[x][1])
      for i in range(len(dict_events_times[trial_time_point_align[x][0]]))]
    
  # print(dict_pho_events_id)  ###################################################################

  return dict_pho_events_id


def get_median_time_and_median_len_by_time_point(dict_events_times, dict_pho_events_id, trial_time_point_align):
  '''Return the median time between events of interest and median of the number of datapoints (length) between these events'''  
  dict_len_time = {x: [] for x in trial_time_point_align}
  dict_len_id = {x: [] for x in trial_time_point_align}
  for et in range(len(dict_events_times)):
    for x in range(len(trial_time_point_align)-1):
      times_e1 = [dict_events_times[et][trial_time_point_align[x][0]][i] + trial_time_point_align[x][1]
                  for i in range(len(dict_events_times[et][trial_time_point_align[x][0]]))]
      times_e2 = [dict_events_times[et][trial_time_point_align[x+1][0]][i] + trial_time_point_align[x+1][1]
                  for i in range(len(dict_events_times[et][trial_time_point_align[x+1][0]]))]
      event1_id = dict_pho_events_id[et][x]
      event2_id = dict_pho_events_id[et][x+1]

      dict_len_time[x].append([e2 - e1 for e1, e2 in zip(times_e1, times_e2)])
      dict_len_id[x].append([e2 - e1 for e1, e2 in zip(event1_id, event2_id)])

  dict_median_time = {x: [] for x in trial_time_point_align}
  dict_median_len = {x: [] for x in trial_time_point_align}
  for x in range(len(trial_time_point_align)-1):
    dict_median_time[x] = int(np.median(np.hstack(dict_len_time[x])))
    dict_median_len[x] = int(np.median(np.hstack(dict_len_id[x])))

  return dict_median_time, dict_median_len

def calculate_latency(dict_events_times):
    '''find events choice_state and stay_in reward poke events. Calculate the latency between these events
    and then scale the values between 0 and 1'''
    latency = []
    scale_latency = []
    for a, b in zip(dict_events_times['choice_state'], dict_events_times['stay']):
        latency.append(b-a)
        
    for i, x in enumerate(latency):
        scale_latency.append((x - min(latency)) / (max(latency) - min(latency)))
        
    return scale_latency

def get_dict_index_session_events(session, event1, event2, possible_events, join_events, join_events2, trial_type):#, **kwargs):
  '''get index in session.events where the events happened
  event1 and event2 are the successive events we want
  possible_events : includes event1 and event2 and all events which could have occurred between these two events.
  '''

  dict_events_id = {}
  dict_events_id['init_trial'], dict_events_id['inter_trial_interval'], _, _ = consecutive_events(
      session, 'init_trial', 'inter_trial_interval', ['init_trial', 'error_time_out', 'inter_trial_interval'])
   
  trials_to_keep = []
  for i in range(len(event1)):
    if event1[i] not in ['stay_left', 'stay_up', 'stay_right']: #stay_in_center or choice_state

        temp_event1_id, temp_event2_id, _, _ = consecutive_events(session, event1[i], event2[i], possible_events[i])
        if event1[i] in dict_events_id: #if choice_state already in the dict
            dict_events_id[event2[i]] = [x for j, x in enumerate(temp_event2_id) if temp_event1_id[j]
                                        in dict_events_id[event1[i]]] #only take the values of stay_in_x if ids of choice_state is already in the dic (which means that the animal comes from the center)
        else:    #if event1 is stay_in_center
            list_event1 = []
            list_event2 = []
            for j in range(len(dict_events_id['init_trial'])):
                for k in range(len(temp_event1_id)):
                    if dict_events_id['init_trial'][j] < temp_event1_id[k] < dict_events_id['inter_trial_interval'][j]:
                        list_event1.append(temp_event1_id[k])
                        list_event2.append(temp_event2_id[k])
            dict_events_id[event1[i]] = list_event1
            dict_events_id[event2[i]] = list_event2

        if event2[i] == 'stay_right': #after all stays have been included in the dictionary, combine them
            for je in join_events: #group stay_in_left, stay_in_up, stay_in_right in a single key 'stay'
              dict_events_id[je] = []
              for x in join_events[je]:
                dict_events_id[je] += dict_events_id[x]
              dict_events_id[je].sort()
        
        
            dict_events_id = {key: dict_events_id[key] for key in ['init_trial', 'inter_trial_interval', 'stay_in_center',
                                                                   'choice_state', 'stay']}            
        
    else: #if we are at the last three events 1 "stay_in"

        temp_event1_id, temp_event2_id, _, _ = consecutive_events(session, event1[i], event2[i], possible_events[i])
        dict_events_id[event2[i]] = [x for j, x in enumerate(temp_event2_id) if temp_event1_id[j]
                                    in dict_events_id['stay']] #only take the values of choose_xxx if ids of stay_in is already in the dict (to have a true sequence of events from the beginning)       
        trials_temp = [j for j, x in enumerate(dict_events_id['stay']) if x
                         in temp_event1_id]
        trials_to_keep.append(trials_temp) #store id of trials with the right sequence of events
    
  for je in join_events2: #group stay_in_left, stay_in_up, stay_in_right in a single key 'stay'
    dict_events_id[je] = []
    for x in join_events2[je]:
      dict_events_id[je] += dict_events_id[x]
    dict_events_id[je].sort()

  dict_events_id = {key: dict_events_id[key] for key in ['init_trial', 'inter_trial_interval', 'stay_in_center',
                                                         'choice_state', 'stay', 'choice']} 
  trials_to_keep = [item for sublist in trials_to_keep for item in sublist] #flatten sublist of trial ids (sublists correspond to each 'stay_in' side)
  trials_to_keep.sort() #put trial ids in order
  
  all_trials = range(len(dict_events_id['init_trial'])) #all trials
  trials_discarded = [x for i, x in enumerate(all_trials) if x not in trials_to_keep]  #store the ids of trial where there is no the right sequence of events and which are not considered

  dict_events_id_true = {'init_trial':[], 'inter_trial_interval':[], 'stay_in_center':[],
                         'choice_state':[], 'stay':[], 'choice':[]} #to have only the events ids corresponding to the trials where there
  #is the right sequence of events

  for key in dict_events_id:
      if key != 'choice':
          for j in range(len(dict_events_id[key])):
              if j in trials_to_keep: #only take the values if trial in trial kept ids
                  dict_events_id_true[key].append(dict_events_id[key][j]) #if true sequence stay_center - choice_state - stay - choice
      elif key == 'choice': #keep the same
          dict_events_id_true[key] = dict_events_id[key]
              
  return dict_events_id_true, trials_to_keep, trials_discarded


def split_data_per_trial(sessions, all_sample_times_pyc, all_corrected_signal, trial_type, time_start, time_end, **kwargs):
    '''Function will return the photometry signals aligned to the different behavioural events
       all_sample_times_pyc: all times of photometry sample in PyControl reference time
       all_corrected_signal: all preprocessed photometry signals
       time_start and time_end: time window before and after the first and last event to extract photometry signals
    '''
    event1 = ['stay_in_center', 'choice_state', 'choice_state', 'choice_state', 'stay_left', 'stay_up', 'stay_right']
    event2 = ['choice_state', 'stay_left', 'stay_up', 'stay_right', 'choose_left', 'choose_up', 'choose_right']
    possible_events = [['stay_in_center', 'same_init_trial', 'choice_state'],
                       ['choice_state', 'stay_left', 'stay_right', 'stay_up','error_time_out'],
                       ['choice_state', 'stay_left', 'stay_right', 'stay_up','error_time_out'],
                       ['choice_state', 'stay_left', 'stay_right', 'stay_up','error_time_out'],
                       ['stay_left', 'grace_left', 'grace_up', 'grace_right', 'choose_left', 'choose_up', 'choose_right'],
                       ['stay_up', 'grace_left', 'grace_up', 'grace_right', 'choose_left', 'choose_up', 'choose_right'],
                       ['stay_right', 'grace_left', 'grace_up', 'grace_right', 'choose_left', 'choose_up', 'choose_right']]  
    
    #all alignement to plot the photometry signals
    trial_time_point_align = {0: ['stay_in_center', time_start], #stay_in_center event minus the start time
                              1: ['stay_in_center', 0], #stay_in_center event
                              2: ['choice_state', 0], #choice_state event
                              3: ['stay', 0], #'stay_in_' reward poke event
                              4: ['choice', 0], # "choose_xxx" event,
                              5: ['choice', time_end]} #'choose_xxx' event plus the end time
    
    join_events = {'stay': ['stay_left', 'stay_up', 'stay_right']}
    join_events2 = {'choice': ['choose_left', 'choose_up', 'choose_right']}

    dict_pho_events_id = []
    dict_events_times = []
    dict_events_id = []
    trials_to_keep_ids = []
    trials_discarded_ids = []
    scale_latency = []
    for session, sample_times_pyc in zip(sessions, all_sample_times_pyc):
        dict_events_id.append(get_dict_index_session_events(session, event1, event2, possible_events, join_events, join_events2, trial_type='all')[0])
        dict_events_times.append(get_dict_event_times(session, dict_events_id[-1]))
        trials_to_keep_ids.append(get_dict_index_session_events(session, event1, event2, possible_events, join_events, join_events2, trial_type='all')[1])        
        trials_discarded_ids.append(get_dict_index_session_events(session, event1, event2, possible_events, join_events, join_events2, trial_type='all')[-1])
        dict_pho_events_id.append(
          get_index_photometry_events(session, sample_times_pyc, dict_events_times[-1], trial_time_point_align))
        
        #store scale latencies from down center poke to reward poke
        scale_latency.append(calculate_latency(dict_events_times[-1]))
        
    dict_median_time, dict_median_len = get_median_time_and_median_len_by_time_point(
        dict_events_times, dict_pho_events_id, trial_time_point_align)
    
    t_scale = {x: [] for x in trial_time_point_align}
    
    pho_scale = {x: [] for x in trial_time_point_align}
    for x in range(len(trial_time_point_align)-1):
      #print(x)
      for s in range(len(sessions)):
        temp_t_scale, temp_pho_scale = zip(*[scale_time_events(
            dict_median_time[x], dict_median_len[x], all_corrected_signal[s], p1, p2)
            for p1, p2 in zip(dict_pho_events_id[s][x], dict_pho_events_id[s][x+1])])
        t_scale[x].append(np.asarray(temp_t_scale))
        pho_scale[x].append(np.asarray(temp_pho_scale))
        print('scaled session {}'.format(s))
    
    if len(t_scale[0]) > 1:  # check if there are more one session to check for correct scaling across sessions
      for x in range(len(t_scale) - 1):
        for i in range(len(t_scale[x])):
          if np.asarray([e == t_scale[x][i][1] for e in np.asarray(t_scale[x][1])]).all() != True:
            raise ValueError('Error in time scaling')
    t_scale = [t_scale[a][0][1] for a in range(len(trial_time_point_align)-1)]

    # print(t_scale)
    # print(pho_scale)
    
    return t_scale, pho_scale, trials_to_keep_ids, trials_discarded_ids, scale_latency


def join_trial_scaling_t_pho_scale(t_scale, pho_scale):
  v_line = []
  for i in range(1, len(t_scale)):
    t_scale[i] = [x + t_scale[i - 1][-1] for x in t_scale[i]] #make times from different couple of events successive in the array
    v_line.append(t_scale[i - 1][-1])
    
  t_scale_whole = np.hstack(t_scale)
  #print('t_whole', t_scale_whole)
  pho_scale_together = pho_scale[0]
  for i in range(1,len(pho_scale)-1):
    pho_scale_together = [np.hstack((x, y)) for x,y in zip(pho_scale_together, pho_scale[i])] #returns a list of lists corresponding to all sessions

  return t_scale_whole, pho_scale_together, v_line


def get_scaled_photometry_per_trial(sessions, all_sample_times_pyc, all_corrected_signal, time_start, time_end):
  '''t_scale : for each period between events, the time after scaling everything to a common time (using medians)
     pho_scale: the corresponding scaled  photometry signals for each period between events
     returns t_scale_whole : all times between events made consecutive
             v_line: corresponds to the times between each periods between events
             trials_to_keep_ids : ids of trials where there the animal actually did the sequence of events as shown in the plot
             z_score_signal : z scored photometry aligned photometry signals
             scale_latency: scaled latency to move from the init poke to reward poke, used for regression
  '''
  t_scale, pho_scale, trials_to_keep_ids, trials_discarded_ids, scale_latency = split_data_per_trial(sessions, all_sample_times_pyc, all_corrected_signal, trial_type='all',
                                        time_start=time_start, time_end=time_end)

  t_scale_whole, pho_scale_together, v_line = join_trial_scaling_t_pho_scale(t_scale, pho_scale)
  
  #transform to z-score
  z_score = [stats.zscore(pho_scale_together[i], axis=None, ddof=0) for i in
                    range(len(pho_scale_together))]

  z_score_signal = [z_score[i].tolist() for i in range(len(z_score))] #list use to insert a false value [0] at trial
  #indices where there are no true value (trial without the sequence of events), to match real number of trials
  for i in range(len(z_score_signal)):
      for j in range(len(trials_discarded_ids[i])):
          z_score_signal[i].insert(trials_discarded_ids[i][j],[0])

  for i in range(len(scale_latency)): #same for latency, for regression
      for j in range(len(trials_discarded_ids[i])):
          scale_latency[i].insert(trials_discarded_ids[i][j],0)
  
  return t_scale_whole, v_line, trials_to_keep_ids, z_score_signal, scale_latency


def find_max_list(list_):
    list_len = [len(i) for i in list_ if type(i) == np.ndarray]
    return(max(list_len))

#def plot_photometry(sessions, all_sample_times_pyc, all_corrected_signal, all_photo_data, all_trial_type, start_str, time_start, time_end):
def plot_photometry(sessions, all_photo_data, z_score_signal, t_scale_whole, v_line, trials_to_keep_ids, all_trial_type, start_str, time_start, time_end):

    '''plot z-score photometry data with signals warped when the time is mouse dependent (at 'GO' signal) '''
    
    subjects = list(set([all_photo_data[i]['subject_ID'] for i in range(len(all_photo_data))]))
    print(subjects)
    
    colors = ['C1', 'C1', 'C1', 'C1', 'C1'] #for dMSN
    #colors = ['C0', 'C0', 'C0', 'C0', 'C0'] # for iMSN

    if start_str == 'A': 
        colors_list = ['maroon', 'red','coral']
    elif start_str == 'D':
        colors_list = ['blue', 'blueviolet', 'fuchsia']
        
    for z in range(len(all_trial_type)): #loop over the list of lists
        all_pho_scale = [] #to have plots on the same figure
        all_pho_sem = []
        for c, trial_type in enumerate(all_trial_type[z]): #for each trial type
          all_sub_pho_scale_mean = [] 
          all_sub_pho_scale_sem = []
          for sub in subjects:
            idx_sub = np.where([all_photo_data[i]['subject_ID'] == sub for i in range(len(all_photo_data))])[0]
            sessions_sub = [sessions[x] for x in idx_sub]
            print(sessions_sub)
            region = [all_photo_data[x]['region'] for x in idx_sub]
            print(region)
            hemisphere = [all_photo_data[x]['hemisphere'] for x in idx_sub]
            #pho_scale_together_sub = [z_score[x] for x in idx_sub]
            pho_scale_together_sub = [z_score_signal[x] for x in idx_sub]
            real_sequence_ids_sub = [trials_to_keep_ids[x] for x in idx_sub]
            
            trial_id_sessions = get_id_trials_to_analyse_all_sessions(sessions_sub, trial_type, region, hemisphere) #take the
            #trials ids corresponding to the trial type specified
            
            if type(trial_id_sessions) is not list:
                trial_id_sessions = [arr.tolist() for arr in trial_id_sessions]


            real_trial_id_sessions = [[trial_id for trial_id in trial_ids if trial_id in real_sequence_ids_sub[i]]
                                      for i, trial_ids in enumerate(trial_id_sessions)] #keep the trial type ids only if they are in the list of trials where there was a true sequence of events
            
            #real_trial_id_sessions = [[] for _ in range(len(trial_id_sessions))] #list of lists to store the ids of trials in which there is really sequence of
            #center poke - choice_state - stay and choice (cue)
        
            #for x in range(len(trial_id_sessions)):
            #    for i in range(len(trial_id_sessions[x])):
            #        if trial_id_sessions[x][i] in real_sequence_ids_sub[x]: #keep the trial type ids only if they are in the list of trials where there was a true sequence of events
            #            real_trial_id_sessions[x].append(trial_id_sessions[x][i])
            
           #only take the signals if the trial was a true succession of events center-out-side
            pho_scale_trial_type = np.asarray([pho_scale_together_sub[s][x] for s in range(len(pho_scale_together_sub))
                                             for x in real_trial_id_sessions[s]])    



            all_sub_pho_scale_mean.append(np.nanmean(pho_scale_trial_type, axis=0))
            all_sub_pho_scale_sem.append(stats.sem(pho_scale_trial_type, axis=0, nan_policy='omit'))
            
          #if can case there is no trials 20% sometimes ?? fill with nans:
          max_list = find_max_list(all_sub_pho_scale_mean)
          for i in range(len(all_sub_pho_scale_mean)):
              if type(all_sub_pho_scale_mean[i]) != np.ndarray:
                  all_sub_pho_scale_mean[i] = [np.nan for i in range(max_list)]
    
          for i in range(len(all_sub_pho_scale_sem)):
              if type(all_sub_pho_scale_sem[i]) != np.ndarray:
                  all_sub_pho_scale_sem[i] = [np.nan for i in range(max_list)]
                        
          #time_start = 1500    
          y = np.nanmean(all_sub_pho_scale_mean, axis=0)
          t = t_scale_whole - time_start
          
          fig, ax = plt.subplots()
          ax.plot(t, y, color=colors[c])
          
          if len(subjects) > 1:
            sem = stats.sem(all_sub_pho_scale_mean, axis=0, nan_policy='omit')
          else:
            sem = all_sub_pho_scale_sem[0] 
            
          ax.fill_between(t, y + sem,
                           y - sem, alpha=0.5, color=colors[c])    
          plt.title(trial_type)
          #print(v_line)
          #plt.xlabel('Time (ms)')
          plt.ylabel('Z-score')
          
          all_pho_scale.append(y)
          all_pho_sem.append(sem)
          
          #plt.ylim([-1, 2.0])
          v_line_param = {0: [[v_line[0] - time_start, 'Init', 140, 'k']], #center poke, initiation
                          1: [[v_line[1] - time_start, 'Go', 140, 'k'], #choice state, side poke illuminate
                              [v_line[2] - time_start, 'Poke', 240, 'k']], #stay in side poke
                          2: [[v_line[3] - time_start, 'Cue', 210, 'k'], #cue
                              [v_line[3] - time_start + 501, 'O', 80, 'darkgrey']]} #outcome
          ax.axes.xaxis.set_visible(False)
        
          [ax.axvline(v_line_param[x][i][0], color='k', ls='--', lw=1) for x in range(len(v_line_param)) for i in range(len(v_line_param[x]))]
          [ax.text(v_line_param[x][i][0], -.05, v_line_param[x][i][1], color='k', fontsize=20, transform=ax.get_xaxis_transform(), rotation='horizontal') for x in range(len(v_line_param)) for i in range(len(v_line_param[x]))] 
        
        #plt.legend(lines, loc='upper center', bbox_to_anchor=(0.3, -0.1), fancybox=False, shadow=False, ncol=1)
        plt.gcf().set_tight_layout(True) 
        
        ##Plots of 3 trial types on the same figure
        
        #change resolution parameters for FENS Poster##  
        plt.rcParams['figure.dpi'] = 300
        plt.rcParams['savefig.dpi'] = 300  
        # change the default font family
        plt.rcParams.update({'font.family':'Arial'})
    
        fig, ax = plt.subplots()
        ax.axes.xaxis.set_visible(False)
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)
        ax.tick_params(axis='y', which='major', labelsize=16)
        [ax.axvline(v_line_param[x][i][0], color=v_line_param[x][i][3], ls='--', lw=1) for x in range(len(v_line_param)) for i in range(len(v_line_param[x]))]
        [ax.text(v_line_param[x][i][0] - v_line_param[x][i][2], -.08, v_line_param[x][i][1], color=v_line_param[x][i][3], fontsize=20, transform=ax.get_xaxis_transform(), rotation='horizontal') for x in range(len(v_line_param)) for i in range(len(v_line_param[x]))] 
        
        line1, = ax.plot(t_scale_whole - time_start, all_pho_scale[0], color=colors_list[0], label=all_trial_type[z][0], linewidth=4)
        ax.fill_between(t_scale_whole - time_start, all_pho_scale[0] + all_pho_sem[0], all_pho_scale[0] - all_pho_sem[0], color=colors_list[0], alpha=0.1)
        
        line2, = ax.plot(t_scale_whole - time_start, all_pho_scale[1], color=colors_list[1], label=all_trial_type[z][1], linewidth=4)
        ax.fill_between(t_scale_whole - time_start, all_pho_scale[1] + all_pho_sem[1], all_pho_scale[1] - all_pho_sem[1], color=colors_list[1], alpha=0.1)
        
        #commented for plotting reward and non reward trials
        line3, = ax.plot(t_scale_whole - time_start, all_pho_scale[2], color=colors_list[2], label=all_trial_type[z][2], linewidth=4)
        ax.fill_between(t_scale_whole - time_start, all_pho_scale[2] + all_pho_sem[2], all_pho_scale[2] - all_pho_sem[2], color=colors_list[2], alpha=0.1)
        ax.legend(handles=[line1, line2, line3], fontsize=16, bbox_to_anchor=(1.1, 0.2, 0.5, 0.5))
        plt.ylabel('Z-score', fontsize=18)        
        
        #Reminder colors
        #for D2 : maroon, red, coral (side and proba) // orange, sienna (reward)
        #for D1: blue, blueviolet, fuchsia (side and proba) // cornflowerblue, midnightblue (reward)
        


#########################################################################################################################
#########################################################################################################################
####################################################REGRESSION###########################################################
#########################################################################################################################
#########################################################################################################################
def _lag(x, i):  # Apply lag of i trials to array x.
  if type(x) == list:
      x = np.array(x)
      
  x_lag = np.zeros(x.shape, x.dtype)
  if i > 0:
      x_lag[i:] = x[:-i]
  else:
      x_lag[:i] = x[-i:]
  return x_lag

def _get_data_to_analyse(session, region='DLS', hemisphere='L'):
  '''extract meaningful data from the session and returns it so can be used then to set predictors for regression
  For example, choices, outcomes, proba trial type, free or forced trials etc...'''
  
  all_event_print = session.events_and_print
  all_prints = [all_event_print[i].name for i in range(len(all_event_print)) if type(all_event_print[i].name) == list and 'T' in all_event_print[i].name[0]] #take only all the print lines appearing at the end of a trial
  trial_type = [all_prints[i][4].split(':')[1] for i in range(len(all_prints))] #all trial type, 'FC'
  couple = [all_prints[i][5].split(':')[1] for i in range(len(all_prints))] #to have the couple offered (None during forced trials and L-R, L-U or U-R during free trials)

  init_id, ITI_id, _, _ = consecutive_events(session, 'init_trial', 'inter_trial_interval', ['init_trial', 'error_time_out', 'inter_trial_interval'])    
  all_init_breaks_id = [i for i in range(len(session.events)) if session.events[i].name == 'same_init_trial']
  trial_with_init_breaks = [1 if [j for j in all_init_breaks_id if init_id[i] < j < ITI_id[i]] else 0 for i in range(len(init_id))] #1 if at least one break in that trial, otherwise 0

  proba_choosed = session.trial_data['proba_choosed']
  choices = session.trial_data['choices'] # 1 - left; 2-up; 3- right
  outcomes = session.trial_data['outcomes'].astype(bool)
  #proba_choosed = list(map(float, [all_prints[i][8].split(':')[1] for i in range(len(all_prints))])) #all proba choosed
  
  forced_choice_trials = ~session.trial_data['free_choice']
  #free_trial = np.where(forced_choice_trials == 0)[0] #ids of free trials 

  lateral_couple = [1 if c == 'L-R' else 0 for c in couple] #1 if the pokes offered during free trial are left and right
  up_right_couple = [1 if c == 'L-U' else 0 for c in couple]
  up_left_couple = [1 if c == 'U-R' else 0 for c in couple]
  
  probas = []
  store_probas = session.store_probas
  for i in range(len(store_probas)): #have a list of probas (float)
      probas.append(float(store_probas[i]))
     
  high_prob = []
  low_prob = []
  for i,x in enumerate(trial_type):
      if x == 'FC':
          high_prob.append(float(all_prints[i][6].split(':')[1])) #if free choice trials, high prob is a float
          low_prob.append(float(all_prints[i][7].split(':')[1]))
      else:
          high_prob.append(str(all_prints[i][6].split(':')[1])) #if free choice trials, high prob is a string ('None')
          low_prob.append(str(all_prints[i][7].split(':')[1]))

  high_proba = [float(all_prints[i][6].split(':')[1]) if x == 'FC' else proba_choosed[i] for i, x in enumerate(trial_type)]          
  index_high_choice = [probas.index(x) + 1 for i, x in enumerate(high_proba)] #high choice index is the index of the highest proba of the couple presented (+1 to match the choices labels)
  
  #extract the option non presented in the free choice trials  
  proba_nonpresented = []
  for i, x in enumerate(trial_type):
      if x == 'FC':
          for x in probas:
              if x not in high_prob and low_prob:
                  proba_nonpresented.append(x)
      else:
          proba_nonpresented.append(0) #in forced trials two choices are non presented, use them for another predictor?           
  #question is how to use this non presented option; as non-presented mean and variance?                
  
  repeat_forced_trial = []
  for i in range(1, len(trial_type)):
      if trial_type[i] in ['L', 'U', 'R']:
          if trial_type[i] == trial_type[i-1]:
              repeat_forced_trial.append(1)
          else:
              repeat_forced_trial.append(-1)    
      else:
          repeat_forced_trial.append(0)
  repeat_forced_trial.insert(0, 0) #0 for the first trial
  
  #ids of free trials where the chosen option at trial t-1 is offered again at trial t
  poke_choosed_repeat_ids = []
  for i in range(1, len(proba_choosed)):
      if trial_type[i-1] == 'FC' and trial_type[i] == 'FC':
          if proba_choosed[i-1] == high_prob[i] or proba_choosed[i-1] == low_prob[i]:
              poke_choosed_repeat_ids.append(i)

  same_free_choice = [] #store ids of free trials where the animal choose the same option than trial t-1
  diff_free_choice = []
  for i in range(1, len(proba_choosed)):
      if trial_type[i-1] == 'FC' and trial_type[i] == 'FC':
          if proba_choosed[i] == proba_choosed[i-1]:
              same_free_choice.append(i)
          elif proba_choosed[i] != proba_choosed[i-1]:
              diff_free_choice.append(i)
              
  #ids of free trials where the animal do not choose the poke that is presented again
  repeat_notchoosed_poke = []
  for i in range(len(choices)):
      if i in poke_choosed_repeat_ids:
          if i in diff_free_choice:
              repeat_notchoosed_poke.append(i)

  same_diff_choice_represented = []
  for i in range(len(choices)):
      if trial_type[i] in ['L', 'U', 'R']: 
          same_diff_choice_represented.append(0)
      elif i in same_free_choice:
          same_diff_choice_represented.append(1) 
      elif i in repeat_notchoosed_poke:
          same_diff_choice_represented.append(-1)
      else:
          same_diff_choice_represented.append(0)

          
  return choices, outcomes, probas, forced_choice_trials, index_high_choice, proba_choosed,\
         hemisphere, repeat_forced_trial, same_diff_choice_represented, lateral_couple, trial_with_init_breaks

def _get_predictors(data_to_analyse, base_predictors, scale_lat, session=[], region=[]):
  '''
  :param base_predictors: list of predictors of interest
  :return: array with all predictors in columns and trials in rows
  '''

  choices, outcomes, probas, forced_choice_trials, index_high_choice, proba_choosed, hemisphere,\
      repeat_forced_trial, same_diff_choice_represented, lateral_couple, trial_with_init_breaks = data_to_analyse

  all_predictors = base_predictors

  n_predictors = len(all_predictors)
  
  choices_l1 = _lag(choices, 1)
  choices_f1 = _lag(choices, -1)  
  reward_l1 = _lag(outcomes, 1)
  reward_l2 = _lag(outcomes, 2)
  reward_l3 = _lag(outcomes, 3)
  reward_f1 = _lag(outcomes, -1)
  
  proba_choosed_l1 = _lag(proba_choosed, 1)
  
  same_ch = choices == choices_l1
  
  bp_values = {}

  for p in base_predictors:
      
    if p == 'init_breaks': #0.5 if previous breaks in initiation port in that trial
      bp_values[p] = np.asarray([0.5 if b == 1 else -0.5 for b in (trial_with_init_breaks * 1)])

    if p == 'left':  # 0.5 for going left
      bp_values[p] = np.asarray([0.5 if c == 1 else 0 for c in (choices)])

    elif p == 'up':  # 0.5 for going up
      bp_values[p] = np.asarray([0.5 if c == 2 else -0.5 for c in (choices)])

    elif p == 'right':  # 0.5 when going right
      bp_values[p] = np.asarray([0.5 if c == 3 else 0 for c in (choices)])

    elif p == 'trials_100':  # 0.5 for going 100
      bp_values[p] = np.asarray([0.5 if c == 1 else 0 for c in (proba_choosed)])

    elif p == 'trials_50':  # 0.5 when going 50
      bp_values[p] = np.asarray([0.5 if c == 0.5 else 0 for c in (proba_choosed)])

    elif p == 'trials_20':  # 0.5 when going 20
      bp_values[p] = np.asarray([0.5 if c == 0.2 else 0 for c in (proba_choosed)])
     
    elif p == 'ipsi_x_trial_100':
      trial_type = [0.5 if c == 1 else 0 for c in (proba_choosed)]
      hemisphere_param = [1 if hemisphere=='L' else -1][0]
      ipsi_choice = [1 * hemisphere_param if c == 1 else -1 * hemisphere_param if c == 3
                      else 0 for c in (choices * 1)]
      bp_values[p] = [t * i for t, i in zip(trial_type, ipsi_choice)]      

    elif p == 'ipsi_x_trial_50':
      trial_type = [0.5 if c == 0.5 else 0 for c in (proba_choosed)]
      hemisphere_param = [1 if hemisphere=='L' else -1][0]
      ipsi_choice = [1 * hemisphere_param if c == 1 else -1 * hemisphere_param if c == 3
                      else 0 for c in (choices * 1)]
      bp_values[p] = [t * i for t, i in zip(trial_type, ipsi_choice)]

    elif p == 'ipsi_x_trial_20':
      trial_type = [0.5 if c == 0.2 else 0 for c in (proba_choosed)]
      hemisphere_param = [1 if hemisphere=='L' else -1][0]
      ipsi_choice = [1 * hemisphere_param if c == 1 else -1 * hemisphere_param if c == 3
                      else 0 for c in (choices * 1)]
      bp_values[p] = [t * i for t, i in zip(trial_type, ipsi_choice)]

    elif p == 'up_x_trial_100':
      trial_type = [0.5 if c == 1 else 0 for c in (proba_choosed)]
      up_choice = [1 if c == 2 else -1 for c in (choices)]
      bp_values[p] = [t * c for t, c in zip(trial_type, up_choice)]

    elif p == 'up_x_trial_50':
      trial_type = [0.5 if c == 0.5 else 0 for c in (proba_choosed)]
      up_choice = [1 if c == 2 else -1 for c in (choices)]
      bp_values[p] = [t * c for t, c in zip(trial_type, up_choice)]
      
    elif p == 'up_x_trial_20':
      trial_type = [0.5 if c == 0.2 else 0 for c in (proba_choosed)]
      up_choice = [1 if c == 2 else -1 for c in (choices)]
      bp_values[p] = [t * c for t, c in zip(trial_type, up_choice)]
      
    elif p == 'mean_probas':        
      #The mean of a Bernoulli distribution is E[X] = p and the variance, Var[X] = p(1-p).
      bp_values[p] = [1 if c == 1 else 0.5 if c == 0.5 else 0.2 for c in (proba_choosed)]     

    elif p == 'variance_probas':     
      #The mean of a Bernoulli distribution is E[X] = p and the variance, Var[X] = p(1-p).
      bp_values[p] = [0 if c == 1 else 0.25 if c == 0.5 else 0.16 for c in (proba_choosed)]        

    elif p == 'centered_mean':
      mean = [1 if c == 1 else 0.5 if c == 0.5 else 0.2 for c in (proba_choosed)]
      mean_mean = np.nanmean(mean)
      bp_values[p] = [i - mean_mean for i in mean]       
        

    elif p == 'centered_variance':
      variance = [0 if c == 1 else 0.25 if c == 0.5 else 0.16 for c in (proba_choosed)]
      mean_var = np.nanmean(variance)
      bp_values[p] = [i - mean_var for i in variance]        
        
    elif p == 'last_mean_probas':# proba's mean in previous trial
      bp_values[p] = [1 if c == 1 else 0.5 if c == 0.5 else 0.2 for c in (proba_choosed_l1)]        
        
    elif p == 'last_variance_probas': #proba's variance in previous trial
      bp_values[p] = [0 if c == 1 else 0.25 if c == 0.5 else 0.16 for c in (proba_choosed_l1)]

    elif p == 'previous_reward_x_trial_50':  # if previous 50% trial was rewarded
      bp_values[p] = np.asarray([0.5 if (r1 == 1 and t == 0.5) else -0.5 if (r1 == 0 and t == 0.5) else 0 for r1, t in zip(reward_l1 * 1, proba_choosed_l1)])  
      bp_values[p][0] = 0
      
    elif p == 'mean_probas_1': #'staircase' coding of predictors (http://www.regorz-statistik.de/en/regression_ordinal_predictor.html)
      mean_level = [3 if c == 1 else 2 if c == 0.5 else 1 for c in proba_choosed]
      bp_values[p] = [0.5 if m > 1 else -0.5 for m in mean_level]

    elif p == 'mean_probas_2': #'staircase' coding of predictors
      mean_level = [3 if c == 1 else 2 if c == 0.5 else 1 for c in proba_choosed]
      bp_values[p] = [0.5 if m > 2 else -0.5 for m in mean_level]

    elif p == 'variance_probas_1':
      variance_level = [0 if c == 1 else 0.25 if c == 0.5 else 0.16 for c in (proba_choosed)]
      bp_values[p] = [0.5 if m > 0 else 0 for m in variance_level]

    elif p == 'variance_probas_2':
      variance_level = [0 if c == 1 else 0.25 if c == 0.5 else 0.16 for c in (proba_choosed)]
      bp_values[p] = [0.5 if m > 0.16 else 0 for m in variance_level]     

    elif p == 'ipsixvariance_probas_1':
      variance_level = [0 if c == 1 else 0.25 if c == 0.5 else 0.16 for c in (proba_choosed)]
      pred_var_level = [0.5 if m > 0 else 0 for m in variance_level]
      
      hemisphere_param = [1 if hemisphere=='L' else -1][0]
      ipsi_choice = [1 * hemisphere_param if c == 1 else -1 * hemisphere_param if c == 3
                      else 0 for c in (choices * 1)]
      bp_values[p] = [v * i for v, i in zip(pred_var_level, ipsi_choice)]

    elif p == 'ipsixvariance_probas_2':
      variance_level = [0 if c == 1 else 0.25 if c == 0.5 else 0.16 for c in (proba_choosed)]
      pred_var_level = [0.5 if m > 0.16 else 0 for m in variance_level]
      
      hemisphere_param = [1 if hemisphere=='L' else -1][0]
      ipsi_choice = [1 * hemisphere_param if c == 1 else -1 * hemisphere_param if c == 3
                      else 0 for c in (choices * 1)]
      bp_values[p] = [v * i for v, i in zip(pred_var_level, ipsi_choice)]

    elif p == 'upxvariance_probas_1':
      variance_level = [0 if c == 1 else 0.25 if c == 0.5 else 0.16 for c in (proba_choosed)]
      pred_var_level = [0.5 if m > 0 else 0 for m in variance_level]
      
      up_choice = [1 if c == 2 else -1 for c in (choices)]
      bp_values[p] = [v * c for v, c in zip(pred_var_level, up_choice)]

    elif p == 'upxvariance_probas_2':
      variance_level = [0 if c == 1 else 0.25 if c == 0.5 else 0.16 for c in (proba_choosed)]
      pred_var_level = [0.5 if m > 0.16 else 0 for m in variance_level]
      
      up_choice = [1 if c == 2 else -1 for c in (choices)]
      bp_values[p] = [v * c for v, c in zip(pred_var_level, up_choice)]

    elif p == 'ipsi x variance': #interaction between direction and variance
      hemisphere_param = [1 if hemisphere=='L' else -1][0]
      ipsi_choice = [1 * hemisphere_param if c == 1 else -1 * hemisphere_param if c == 3
                      else 0 for c in (choices * 1)]
      variance_ = [0 if c == 1 else 0.25 if c == 0.5 else 0.16 for c in (proba_choosed)]
      bp_values[p] = [i * v for i, v in zip(ipsi_choice, variance_)] 

    elif p == 'up x variance':
      up_choice = [1 if c == 2 else -1 for c in (choices)]
      variance_ = [0 if c == 1 else 0.25 if c == 0.5 else 0.16 for c in (proba_choosed)]
      bp_values[p] = [u * v for u, v in zip(up_choice, variance_)]
      
    elif p == 'ipsi x mean':
      hemisphere_param = [1 if hemisphere=='L' else -1][0]
      ipsi_choice = [1 * hemisphere_param if c == 1 else -1 * hemisphere_param if c == 3
                      else 0 for c in (choices * 1)]
      mean_ = [1 if c == 1 else 0.5 if c == 0.5 else 0.2 for c in (proba_choosed)]
      bp_values[p] = [i * m for i, m in zip(ipsi_choice, mean_)] 

    elif p == 'contra x centered_mean':
      #Thomas Akam's suggestion, centered the regressor to avoid colinearity with the other mean regressor  
      hemisphere_param = [1 if hemisphere=='L' else -1][0]
      contra_choice = [-0.5 * hemisphere_param if c == 1 else 0.5 * hemisphere_param if c == 3 
                      else 0 for c in (choices)]
      mean_ = [1 if c == 1 else 0.5 if c == 0.5 else 0.2 for c in (proba_choosed)]
      mean_mean = np.nanmean(mean_)
      centered_mean = [i - mean_mean for i in mean_]
      bp_values[p] = [i * m for i, m in zip(contra_choice, centered_mean)]              

    elif p == 'contra x centered_variance':
      #Thomas Akam's suggestion, centered the regressor to avoid colinearity with the other mean regressor  
      hemisphere_param = [1 if hemisphere=='L' else -1][0]
      contra_choice = [-0.5 * hemisphere_param if c == 1 else 0.5 * hemisphere_param if c == 3 
                      else 0 for c in (choices)]
      variance_ = [0 if c == 1 else 0.25 if c == 0.5 else 0.16 for c in (proba_choosed)]
      mean_variance = np.nanmean(variance_)
      centered_var = [i - mean_variance for i in variance_]
      bp_values[p] = [i * m for i, m in zip(contra_choice, centered_var)]

    elif p == 'up x centered_mean':
      up_choice = [1 if c == 2 else -1 for c in (choices)]
      mean_ = [1 if c == 1 else 0.5 if c == 0.5 else 0.2 for c in (proba_choosed)]
      mean_mean = np.nanmean(mean_)
      centered_mean = [i - mean_mean for i in mean_]
      bp_values[p] = [i * m for i, m in zip(up_choice, centered_mean)]

    elif p == 'up x centered_variance':
      up_choice = [1 if c == 2 else -1 for c in (choices)]
      variance_ = [0 if c == 1 else 0.25 if c == 0.5 else 0.16 for c in (proba_choosed)]
      mean_variance = np.nanmean(variance_)
      centered_var = [i - mean_variance for i in variance_]
      bp_values[p] = [i * m for i, m in zip(up_choice, centered_var)]

    elif p == 'up x mean':
      up_choice = [1 if c == 2 else -1 for c in (choices)]
      mean_ = [1 if c == 1 else 0.5 if c == 0.5 else 0.2 for c in (proba_choosed)]
      bp_values[p] = [i * m for i, m in zip(up_choice, mean_)]      
      
    elif p == 'ipsi x null_variance':
      hemisphere_param = [1 if hemisphere=='L' else -1][0]
      ipsi_choice = [0.5 * hemisphere_param if c == 1 else -0.5 * hemisphere_param if c == 3
                      else 0 for c in (choices * 1)]
      variance_ = [0 if c == 1 else 0.25 if c == 0.5 else 0.16 for c in (proba_choosed)]
      bp_values[p] = [0.5 if (c==0.5 and v==0) else -0.5 if (c==-0.5 and v==0) else 0
                      for c, v in zip(ipsi_choice, variance_)]        

    elif p == 'ipsi x low_variance':
      hemisphere_param = [1 if hemisphere=='L' else -1][0]
      ipsi_choice = [0.5 * hemisphere_param if c == 1 else -0.5 * hemisphere_param if c == 3
                      else 0 for c in (choices * 1)]
      variance_ = [0 if c == 1 else 0.25 if c == 0.5 else 0.16 for c in (proba_choosed)]
      bp_values[p] = [0.5 if (c==0.5 and v==0.16) else -0.5 if (c==-0.5 and v==0.16) else 0
                      for c, v in zip(ipsi_choice, variance_)] 

    elif p == 'ipsi x high_variance':
      hemisphere_param = [1 if hemisphere=='L' else -1][0]
      ipsi_choice = [0.5 * hemisphere_param if c == 1 else -0.5 * hemisphere_param if c == 3
                      else 0 for c in (choices * 1)]
      variance_ = [0 if c == 1 else 0.25 if c == 0.5 else 0.16 for c in (proba_choosed)]
      bp_values[p] = [0.5 if (c==0.5 and v==0.25) else -0.5 if (c==-0.5 and v==0.25) else 0
                      for c, v in zip(ipsi_choice, variance_)] 
      
    elif p == 'ipsi x low_mean':
      hemisphere_param = [1 if hemisphere=='L' else -1][0]
      ipsi_choice = [0.5 * hemisphere_param if c == 1 else -0.5 * hemisphere_param if c == 3
                      else 0 for c in (choices * 1)]
      mean_ = [1 if c == 1 else 0.5 if c == 0.5 else 0.2 for c in (proba_choosed)]
      bp_values[p] = [0.5 if (c==0.5 and m==0.2) else -0.5 if (c==-0.5 and m==0.2) else 0
                      for c, m in zip(ipsi_choice, mean_)]        

    elif p == 'ipsi x med_mean':
      hemisphere_param = [1 if hemisphere=='L' else -1][0]
      ipsi_choice = [0.5 * hemisphere_param if c == 1 else -0.5 * hemisphere_param if c == 3
                      else 0 for c in (choices * 1)]
      mean_ = [1 if c == 1 else 0.5 if c == 0.5 else 0.2 for c in (proba_choosed)]
      bp_values[p] = [0.5 if (c==0.5 and m==0.5) else -0.5 if (c==-0.5 and m==0.5) else 0
                      for c, m in zip(ipsi_choice, mean_)] 

    elif p == 'ipsi x high_mean':
      hemisphere_param = [1 if hemisphere=='L' else -1][0]
      ipsi_choice = [0.5 * hemisphere_param if c == 1 else -0.5 * hemisphere_param if c == 3
                      else 0 for c in (choices * 1)]
      mean_ = [1 if c == 1 else 0.5 if c == 0.5 else 0.2 for c in (proba_choosed)]
      bp_values[p] = [0.5 if (c==0.5 and m==1) else -0.5 if (c==-0.5 and m==1) else 0
                      for c, m in zip(ipsi_choice, mean_)]      

    elif p == 'up x null_variance':
      up_choice = [0.5 if c == 2 else -0.5 for c in (choices)]
      variance_ = [0 if c == 1 else 0.25 if c == 0.5 else 0.16 for c in (proba_choosed)]
      bp_values[p] = [0.5 if (c==0.5 and v==0) else -0.5 if (c==-0.5 and v==0) else 0
                      for c, v in zip(up_choice, variance_)]        

    elif p == 'up x low_variance':
      up_choice = [0.5 if c == 2 else -0.5 for c in (choices)]
      variance_ = [0 if c == 1 else 0.25 if c == 0.5 else 0.16 for c in (proba_choosed)]
      bp_values[p] = [0.5 if (c==0.5 and v==0.16) else -0.5 if (c==-0.5 and v==0.16) else 0
                      for c, v in zip(up_choice, variance_)] 

    elif p == 'up x high_variance':
      up_choice = [0.5 if c == 2 else -0.5 for c in (choices)]
      variance_ = [0 if c == 1 else 0.25 if c == 0.5 else 0.16 for c in (proba_choosed)]
      bp_values[p] = [0.5 if (c==0.5 and v==0.25) else -0.5 if (c==-0.5 and v==0.25) else 0
                      for c, v in zip(up_choice, variance_)] 
      
    elif p == 'up x low_mean':
      up_choice = [0.5 if c == 2 else -0.5 for c in (choices)]
      mean_ = [1 if c == 1 else 0.5 if c == 0.5 else 0.2 for c in (proba_choosed)]
      bp_values[p] = [0.5 if (c==0.5 and m==0.2) else -0.5 if (c==-0.5 and m==0.2) else 0
                      for c, m in zip(up_choice, mean_)]        

    elif p == 'up x med_mean':
      up_choice = [0.5 if c == 2 else -0.5 for c in (choices)]
      mean_ = [1 if c == 1 else 0.5 if c == 0.5 else 0.2 for c in (proba_choosed)]
      bp_values[p] = [0.5 if (c==0.5 and m==0.5) else -0.5 if (c==-0.5 and m==0.5) else 0
                      for c, m in zip(up_choice, mean_)] 

    elif p == 'up x high_mean':
      up_choice = [0.5 if c == 2 else -0.5 for c in (choices)]
      mean_ = [1 if c == 1 else 0.5 if c == 0.5 else 0.2 for c in (proba_choosed)]
      bp_values[p] = [0.5 if (c==0.5 and m==1) else -0.5 if (c==-0.5 and m==1) else 0
                      for c, m in zip(up_choice, mean_)] 

    elif p == 'forced_trials': # 0.5 forced, -0.5 free
      bp_values[p] = np.asarray([0.5 if f == 1 else -0.5 for f in (forced_choice_trials * 1)])        
      bp_values[p] = forced_choice_trials * 1
    
    elif p == 'repeat_forced_trial':
      bp_values[p] = np.asarray([0.5 if r == 1 else -0.5 if r == -1 else 0 for r in (repeat_forced_trial)])  
    
    elif p == 'latency':
      bp_values[p] = np.asarray(scale_lat)  
    
    elif p == 'reward':  # 0.5 rewarded trials, -0.5 non-rewarded trials
      # bp_values[p] = outcomes * 1
      bp_values[p] = np.asarray([0.5 if o == 1 else -0.5 for o in (outcomes * 1)])
      
    elif p == 'cum_rew': # cumulative reward
      bp_values[p] = list(np.cumsum(outcomes))      

    elif p == 'previous_reward':  # 0.5 if reward in the previous trial; -0.5 if not rewarded
      #bp_values[p] = reward_l1 * 1
      bp_values[p] = np.asarray([0.5 if r1 == 1 else -0.5 for r1 in (reward_l1 * 1)])  
      bp_values[p][0] = 0
    
    elif p == 'previous_reward_x_variance': #try this regressor to account from which option's uncertainty last reward was obtained
      variance_level = [0 if c == 1 else 0.25 if c == 0.5 else 0.16 for c in (proba_choosed_l1)]
      previous_reward = [0.5 if r1 == 1 else -0.5 for r1 in (reward_l1 * 1)]
      bp_values[p] = [r1 * v for r1, v in zip(previous_reward, variance_level)]
 
    elif p == 'previous_reward_x_cent_variance': #test to account from which option's uncertainty last reward was obtained
      #with centered values of variance to avoid colinearity
      variance_level = [0 if c == 1 else 0.25 if c == 0.5 else 0.16 for c in (proba_choosed_l1)]
      mean_variance = np.nanmean(variance_level)
      centered_var = [i - mean_variance for i in variance_level]      
      previous_reward = [0.5 if r1 == 1 else -0.5 for r1 in (reward_l1 * 1)]
      bp_values[p] = [r1 * v for r1, v in zip(previous_reward, centered_var)]

    elif p == 'previous_reward_x_mean': #test to account from which option's value last reward was obtained
      mean_level = [1 if c == 1 else 0.5 if c == 0.5 else 0.2 for c in (proba_choosed_l1)]
      previous_reward = [0.5 if r1 == 1 else -0.5 for r1 in (reward_l1 * 1)]
      bp_values[p] = [r1 * v for r1, v in zip(previous_reward, mean_level)]      

    elif p == 'previous_reward_x_cent_mean': #test to account from which option's value last reward was obtained
      #with centered values of mean to avoid colinearity
      mean_level = [1 if c == 1 else 0.5 if c == 0.5 else 0.2 for c in (proba_choosed_l1)]
      mean_val = np.nanmean(mean_level)
      centered_mean = [i - mean_val for i in mean_level]       
      previous_reward = [0.5 if r1 == 1 else -0.5 for r1 in (reward_l1 * 1)]
      bp_values[p] = [r1 * v for r1, v in zip(previous_reward, centered_mean)] 
      
    elif p == 'reward-2':  # 0.5 if reward in the trial t-2; -0.5 if not rewarded
      #bp_values[p] = reward_l1 * 1
      bp_values[p] = np.asarray([0.5 if r2 == 1 else -0.5 for r2 in (reward_l2 * 1)])  
      bp_values[p][0] = 0

    elif p == 'reward-3':  # 0.5 if reward in the trial t-3; -0.5 if not rewarded
      #bp_values[p] = reward_l1 * 1
      bp_values[p] = np.asarray([0.5 if r3 == 1 else -0.5 for r3 in (reward_l3 * 1)])  
      bp_values[p][0] = 0

    elif p == 'reward_rate':
      moving_reward_rate = exp_mov_ave(tau=10, init_value=0.5)
      moving_reward_average_session = []
      for x in reward_l1:
        moving_reward_rate.update(x)
        moving_reward_average_session.append(moving_reward_rate.value)
      # bp_values[p] = np.asarray([1 if x > 0.7 else 0 for x in moving_reward_average_session])
      bp_values[p] = np.asarray(moving_reward_average_session)

    elif p == 'high_proba':
      correct = (choices == index_high_choice)
      bp_values[p] = np.asarray([0.5 if c == True else -0.5 for c in correct])
      
    elif p == 'forced_choice_x_correct': # +0.5 forced correct, -0.5 forced incorrect, 0 free
      correct = (choices == index_high_choice)  
      bp_values[p] = np.asarray([0.5 if ((c == True) and (f == True)) else -0.5 if ((c == False) and (f == True))
                    else 0 for c, f in zip(correct, forced_choice_trials)])
      
    elif p == 'forced_choice_x_uncorrect': # +0.5 forced uncorrect, -0.5 forced incorrect, 0 free
      uncorrect = (choices != index_high_choice)
      bp_values[p] = np.asarray([0.5 if ((c == True) and (f == True)) else -0.5 if ((c == False) and (f == True))
                    else 0 for c, f in zip(uncorrect, forced_choice_trials)])
      
    elif p == 'free_choice_x_correct': # +0.5 free correct, -0.5 free incorrect, 0 forced
      correct = (choices == index_high_choice)     
      bp_values[p] = np.asarray([0.5 if ((c == True) and (f == False)) else -0.5 if ((c == False) and (f == False))
                    else 0 for c, f in zip(correct, forced_choice_trials)])
    
    elif p == 'free_x_mean': #option 1 if want to regroup both forced and free       
      mean = [1 if c == 1 else 0.5 if c == 0.5 else 0.2 for c in (proba_choosed)]
      mean_mean = np.nanmean(mean)
      bp_values[p] = [m - mean_mean if (f == False) else -(m - mean_mean) for m, f in zip(mean, forced_choice_trials)]
      
    #elif p == 'forced_x_mean':        
    #  mean = [1 if c == 1 else 0.5 if c == 0.5 else 0.2 for c in (proba_choosed)]
    #  mean_mean = np.nanmean(mean)
    #  bp_values[p] = [m - mean_mean if (f == True) else 0 for m, f in zip(mean, forced_choice_trials)] 
      
    elif p == 'free_x_variance':        
      variance = [0 if c == 1 else 0.25 if c == 0.5 else 0.16 for c in (proba_choosed)]
      mean_variance = np.nanmean(variance)
      bp_values[p] = [m - mean_variance if (f == False) else -(m - mean_variance) for m, f in zip(variance, forced_choice_trials)]
      
    #elif p == 'forced_x_variance':        
    #  variance_ = [0 if c == 1 else 0.25 if c == 0.5 else 0.16 for c in (proba_choosed)]
    #  mean_variance = np.nanmean(variance_)
    #  bp_values[p] = [m - mean_variance if (f == True) else 0 for m, f in zip(variance_, forced_choice_trials)]      
    #elif p == 'unpresented_free':  

    elif p == 'ipsi_choice': # +0.5 ipsilateral choice, -0.5 contralateral choice
      hemisphere_param = [1 if hemisphere=='L' else -1][0]
      bp_values[p] = [0.5 * hemisphere_param if c == 1 else -0.5 * hemisphere_param if c == 3
                      else 0 for c in (choices * 1)]

    elif p == 'contra_choice': # -0.5 ipsilateral choice, +0.5 contralateral choice
      hemisphere_param = [1 if hemisphere=='L' else -1][0]
      bp_values[p] = [-0.5 * hemisphere_param if c == 1 else 0.5 * hemisphere_param if c == 3 
                      else 0 for c in (choices)]

    elif p == 'previous_contra_choice': # -0.5 ipsilateral choice, +0.5 contralateral choice
      hemisphere_param = [1 if hemisphere=='L' else -1][0]
      bp_values[p] = [-0.5 * hemisphere_param if c == 1 else 0.5 * hemisphere_param if c == 3 
                      else 0 for c in (choices_l1)]

    elif p == 'previous_up_choice': # -0.5 previous up choice, 0.5 other choice
      bp_values[p] = [0.5 if c == 2 else -0.5 for c in (choices_l1)]
      
    #elif p == 'repeat_free_choice': # +0.5 if same choice as previous choice, -0.5 if different choice to previous trial, 0 if forced
    #  same_ch = (choices == choices_l1) * 1
    #  bp_values[p] = np.asarray([0.5 if ((c == 1) and (f == False)) else -0.5 if ((c == 0) and (f == False))
    #                else 0 for c, f in zip(same_ch, forced_choice_trials)])
    
    elif p == 'repeat_selected_free_choice':
      #previous selected poke is represented, code positively if animal poke it again, negative if not, and 0 if not represented or forced trials
      bp_values[p] = np.asarray([0.5 if s == 1 else -0.5 if s == -1 else 0 for s in (same_diff_choice_represented)]) 

    elif p == 'diff_ch_x_norewl1':
      diff_ch = (choices != choices_l1) * 1  
      bp_values[p] = np.asarray([x * 0.5 if (x == 1 and r1 == 0 and f == False) else x * (-0.5) if (x == 1 and r1 == 1 and f==False) else x
                      for x, r1, f in zip(diff_ch, reward_l1, forced_choice_trials)]) 
      
    elif p == 'same_ch_x_norewl1':
      same_ch = (choices == choices_l1) * 1  
      bp_values[p] = np.asarray([x * 0.5 if (x == 1 and r1 == 0 and f == False) else x * (-0.5) if (x == 1 and r1 == 1 and f==False) else x
                      for x, r1, f in zip(same_ch, reward_l1, forced_choice_trials)])      

    elif p == 'diff_future_ch': #+0.5 if next choice is different, -0.5 if next same choice
      future_diff_ch = (choices != choices_f1) * 1     
      bp_values[p] = np.asarray([0.5 if ((c == 1) and (f == False)) else -0.5 if ((c == 0) and (f == False))
                    else 0 for c, f in zip(future_diff_ch, forced_choice_trials)]) 
      bp_values[p][-1] = 0  

  # Generate lagged predictors from base predictors.
  n_trials = len(choices)
  predictors = np.zeros([n_trials, n_predictors])

  for i, p in enumerate(all_predictors):
      bp_name = p
      predictors[:, i] = bp_values[bp_name][:]

  return (choices, predictors)

class exp_mov_ave:
    # Exponential moving average class.
    def __init__(self, tau, init_value=0):
        self.tau = tau
        self.init_value = init_value
        self.reset()

    def reset(self, init_value=None, tau=None):
        if tau:
            self.tau = tau
        if init_value:
            self.init_value = init_value
        self.value = self.init_value
        self._m = math.exp(-1./self.tau)
        self._i = 1 - self._m

    def update(self, sample):
        self.value = (self.value * self._m) + (self._i * sample)


def _get_session_photometry_predictors(session, pho_scale_together_i, base_predictors, trials_to_keep_ids, scale_lat, forced_choice, region, hemisphere):
  '''return the signals and predictors of the trials considered
  Either forced_choice == True : all trials (forced and free) considered
  forced_choice == 'only' : only forced trials considered
  forced_choice == False : only free trials considered
  '''
  data_to_analyse = _get_data_to_analyse(session, region, hemisphere)

  choices, predictors = _get_predictors(data_to_analyse, base_predictors, scale_lat, session, region=region)
  # select trials to analyse
  dict_events_id = select_trial_types_to_analyse(session)  # dict_events_id['all']: positions of the
  # think these are trials not events

  if forced_choice == True:
    #include all trials
    pho_scale = [pho_scale_together_i[x] for x in dict_events_id['all'] if x in trials_to_keep_ids]    
    predictors = [predictors[x] for x in dict_events_id['all'] if x in trials_to_keep_ids]

  elif forced_choice == 'only':
    # analyse only forced choice trials
    #pho_scale = [pho_scale_together_i[x] for x in dict_events_id['all'] if x in trials_to_keep_ids and x not in dict_events_id['free_trial'] and x not in dict_events_id['up_trial']]
    #choices = [choices[x] for x in dict_events_id['all'] if x in trials_to_keep_ids and x not in dict_events_id['free_trial'] and x not in dict_events_id['up_trial']]
    #predictors = [predictors[x] for x in dict_events_id['all'] if x in trials_to_keep_ids and x not in dict_events_id['free_trial'] and x not in dict_events_id['up_trial']]
    pho_scale = [pho_scale_together_i[x] for x in dict_events_id['all'] if x in trials_to_keep_ids and x not in dict_events_id['free_trial']]
    predictors = [predictors[x] for x in dict_events_id['all'] if x in trials_to_keep_ids and x not in dict_events_id['free_trial']]

  else: #elif forced_choice == False
    #eliminate forced choice trials
    pho_scale = [pho_scale_together_i[x] for x in dict_events_id['all'] if x in dict_events_id['free_trial']]
    predictors = [predictors[x] for x in dict_events_id['all'] if x in dict_events_id['free_trial']]

  return pho_scale, predictors

def _compute_photometry_regression(sessions, pho_scale_together, base_predictors, trials_to_keep_ids, scale_lat, forced_choice=False,
                                   region=[], hemisphere=[], regression_type='Linear',
                                   plot_correlation=False, return_predictors_array=False):
  '''
  regression_type = 'Linear'
  '''

  all_predictors = []
  all_pho_scale = []

  for i, session in enumerate(sessions):
    
    pho_scale, predictors = _get_session_photometry_predictors(session=session,
                                                               pho_scale_together_i=pho_scale_together[i],
                                                               base_predictors=base_predictors, trials_to_keep_ids=trials_to_keep_ids[i],
                                                               scale_lat = scale_lat[i],
                                                               forced_choice=forced_choice,
                                                               region=region[i], hemisphere=hemisphere[i])

    if all_predictors != []:
      all_predictors += predictors
      all_pho_scale += pho_scale
    else:
      all_predictors = predictors[:]
      all_pho_scale = pho_scale[:]
  if return_predictors_array is True:
    return all_predictors, all_pho_scale
  
  # Linear regression 
  if regression_type == 'Linear':
    log_reg = lm.LinearRegression()
  idx_nan = np.where(np.isnan(all_pho_scale))[0]
  all_pho_scale = [all_pho_scale[i] for i in range(len(all_pho_scale)) if i not in idx_nan]
  all_predictors = [all_predictors[i] for i in range(len(all_predictors)) if i not in idx_nan]

  log_reg.fit(all_predictors, all_pho_scale) #run the regression
  print(log_reg.score(all_predictors, all_pho_scale))
    
  if plot_correlation == True: #plot a heatmap showing the correlation between all the predictors
    if len(base_predictors) > 1:
      plt.figure()
      plt.imshow(np.corrcoef(np.asarray(all_predictors).T))
      plt.xticks(range(len(base_predictors)), base_predictors, fontsize=6, rotation=90)
      plt.yticks(range(len(base_predictors)), base_predictors, fontsize=6)
      plt.colorbar()
      plt.gcf().set_tight_layout(True)

  corr1 = np.asarray(base_predictors)[np.where(np.corrcoef(np.asarray(all_predictors).T) > 0.7)[0][np.not_equal(
    np.where(np.corrcoef(np.asarray(all_predictors).T) > 0.7)[0], np.where(np.corrcoef(np.asarray(all_predictors).T) > 0.7)[1])]]
  corr2 = np.asarray(base_predictors)[np.where(np.corrcoef(np.asarray(all_predictors).T) > 0.7)[1][np.not_equal(
    np.where(np.corrcoef(np.asarray(all_predictors).T) > 0.7)[0], np.where(np.corrcoef(np.asarray(all_predictors).T) > 0.7)[1])]]

  if corr1.size:
    for icorr in range(len(corr1)):
      warnings.warn('{} and {} are >0.7 correlated'.format(corr1[icorr], corr2[icorr])) #alert if high correlation between two predictors

  return log_reg.coef_, log_reg.intercept_, all_predictors, all_pho_scale

def normalise(x,dim=0):
    return (x-np.mean(x,dim, keepdims=True))/np.std(x,dim, keepdims=True)

def plot_photometry_regression(sessions, all_photo_data, t_scale_whole, pho_scale_together, v_line, time_start, base_predictors,
                               trials_to_keep_ids, scale_latency, forced_choice=True, title=[],
                               all_event_lines=True, plot_legend=True,
                               regression_type='Linear', subplots=False, figsize=(4,25),
                               plot_correlation=False, plot_intercept=True, type_coef='non-norm', test_visual='dots', **kwargs):
  '''
  takes photometry signals (pho_scale_together can be replace by z_scored_signal when the function is called)
  
  scale_latency = scaled latency from init poke to reward poke
  
  forced_choice=True : analysis include all trials
  forced_choice='only' : analysis include only forced trials
  forced_choice=False : exclude forced trials
  
  all_event_lines=True : add lines alignement with behavioral events
  
  subplots=False :  all the regressors curves on the same plot
  subplots= list of lists with subplots : every subplots contain designated regressors
  
  plot_correlation=True : plot correlation matrix with correlation between all regressors with each other for each individual
  
  plot_intercept=True: add the intercept curve in the plot
  
  type_coef='norm' : z_score each regressor coefficients
  type_coef='non-norm': does not z scored the regressors
  
  test_visual='dots': t_test at each timepoint against 0 and plot dots above the curve if regressor coeff is significantly different from 0 
  '''
  
  colors = kwargs.pop('colors', ['C0', 'darkred', 'C2', 'C3', 'C4', 'C5', 'C6', 'indigo', 'C8', 'C9', 'b', 'orange', 'C0', 'black', 'darkred',
            'g', 'crimson', 'aqua', 'C0', 'b', 'C2', 'C0', 'darkred', 'C2', 'black', 'indigo', 'C8', 'C9', 'b', 'orange', 'gold', 'black'])
  subjects_list = kwargs.pop('subjects_list', False)
  per_subject = kwargs.pop('per_subject', True)

  if subjects_list == False:
    subjects = list(set([all_photo_data[i]['subject_ID'] for i in range(len(all_photo_data))]))
  else:
    subjects = ['{}'.format(sl) for sl in subjects_list]
    subjects = [sub for sub in subjects if sub in list(set([all_photo_data[i]['subject_ID'] for i in range(len(all_photo_data))]))]
  coef_reg_all = []
  if per_subject == False:
    subjects = [subjects]
  for sub in subjects:
    print(sub)
    idx_sub = np.where([all_photo_data[i]['subject_ID'] in sub for i in range(len(all_photo_data))])[0]
    sessions_sub = [sessions[x] for x in idx_sub]
    region = [all_photo_data[x]['region'] for x in idx_sub]
    hemisphere = [all_photo_data[x]['hemisphere'] for x in idx_sub]
    pho_scale_together_sub = [pho_scale_together[x] for x in idx_sub]
    trials_to_keep_ids_sub = [trials_to_keep_ids[x] for x in idx_sub]
    scale_lat_sub = [scale_latency[x] for x in idx_sub]

    if regression_type != 'OLS':
      #print(sessions_sub, forced_choice, region, regression_type, plot_correlation)
      coef_reg, coef_intercept, all_predictors, all_pho_scale = _compute_photometry_regression(sessions=sessions_sub,
                                                                 pho_scale_together=pho_scale_together_sub,
                                                                 base_predictors=base_predictors,
                                                                 trials_to_keep_ids=trials_to_keep_ids_sub,
                                                                 scale_lat=scale_lat_sub,
                                                                 forced_choice=forced_choice,
                                                                 region=region, hemisphere=hemisphere,regression_type=regression_type,
                                                                 plot_correlation=plot_correlation)
      
      df = pd.DataFrame(coef_reg) #create a pandas dataframe, only useful if we want z-scored coefficients
      if plot_intercept == True: #add intercept coefficients to the other coeff
        if type_coef == 'norm': #normalize (zscore) all coefficients individually
            df[len(df.columns)] = coef_intercept #add column intercept
            df_zscore = df.apply(stats.zscore)
            array_coef = df_zscore.to_numpy() #change back to numpy array
            coef_reg_all.append(array_coef)
        elif type_coef == 'non_norm':
            coef_reg_all.append(np.append(coef_reg, [[x] for x in coef_intercept], axis=1))
            
      elif plot_intercept == False: 
        if type_coef == 'norm':
            df.apply(stats.zscore)
            array_coef = df.to_numpy()
            coef_reg_all.append(array_coef)
        elif type_coef == 'non_norm':
            coef_reg_all.append(coef_reg)

  if plot_intercept == True:
    base_predictors = base_predictors + ['intercept']

  if (len(subjects) == 1) or (per_subject == False):
    mean = coef_reg_all[0]
    sem = np.nan
  else:
    mean = np.mean(coef_reg_all, axis=0)
    sem = stats.sem(coef_reg_all, axis=0)

  #t-test  
  t, prob = stats.ttest_1samp([coef_reg_all[i][::1] for i in range(len(coef_reg_all))], 0, axis=0)
  
  v_line_param = {0: [[v_line[0] - time_start, 'Init', -0.01]], #center poke, initiation
                  1: [[v_line[1] - time_start, 'Go', 0.02], #choice state, side poke illuminate
                      [v_line[2] - time_start, 'Poke', -0.01]], #stay in side poke
                  2: [[v_line[3] - time_start, 'Cue', -0.01], #cue
                      [v_line[3] - time_start + 501, 'O', -0.01]]} #outcome
    
  #change resolution parameters for FENS Poster##  
  plt.rcParams['figure.dpi'] = 300
  plt.rcParams['savefig.dpi'] = 300  
  # change the default font family
  plt.rcParams.update({'font.family':'Arial'})  

  if subplots == False: #all coefficients plots on the same figure
    fig, ax = plt.subplots()
    for i in range(len(base_predictors)):
      ax.plot(t_scale_whole-time_start, mean.T[i], color=colors[i], label=base_predictors[i])

    if all_event_lines == True:
      ax.axes.xaxis.set_visible(False)
      [ax.axvline(v_line_param[x][i][0], color='k', lw=0.5) for x in range(len(v_line_param)) for i in range(len(v_line_param[x]))]
      [ax.text(v_line_param[x][i][0], -.05, v_line_param[x][i][1], color='k', transform=ax.get_xaxis_transform(), rotation='horizontal') for x in range(len(v_line_param)) for i in range(len(v_line_param[x]))] 
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.spines['left'].set_linewidth(2)
    ax.spines['bottom'].set_linewidth(2)
    
    plt.ylabel('Regression coefficients')
    plt.xlabel('Time (ms)')
    plt.margins(0, 0.05)
    plt.axhline(0, linestyle='--', color='k')
    if title != []:
      plt.title(title)
    if plot_legend == True:
      plt.legend(loc='lower left', bbox_to_anchor=(0.5,-0.2), fontsize=10)
    plt.gcf().set_tight_layout(True)

  else: #different subplots with one or few coeff plots in each
    fig, axs = plt.subplots(len(subplots), sharex=True, figsize=figsize)
    for sub in range(len(subplots)):
      for sub_pred in subplots[sub]:          
          axs[sub].plot(t_scale_whole-time_start, mean.T[base_predictors.index(sub_pred)], color=colors[base_predictors.index(sub_pred)], label=sub_pred, linewidth=3.5)

          if type_coef == 'non_norm':
              axs[sub].set_ylabel('Regression coefficients', fontsize=16)
          elif type_coef == 'norm':
              axs[sub].set_ylabel('Regression coefficients\n(z-score)', fontsize=16)
             
          if (len(subjects) > 1):                 
              axs[sub].fill_between(t_scale_whole - time_start, mean.T[base_predictors.index(sub_pred)] +
                                    sem.T[base_predictors.index(sub_pred)], mean.T[base_predictors.index(sub_pred)] -
                                    sem.T[base_predictors.index(sub_pred)], alpha=0.12,
                                    facecolor=colors[base_predictors.index(sub_pred)])
                                                
      if all_event_lines == True:
        axs[sub].spines['top'].set_visible(False)
        axs[sub].spines['right'].set_visible(False)
        axs[sub].spines['left'].set_linewidth(2)
        axs[sub].spines['bottom'].set_linewidth(2)
        axs[sub].axes.xaxis.set_visible(False)  
        [axs[sub].axvline(v_line_param[x][j][0], color='k',lw=0.5) for x in range(len(v_line_param)) for j in range(len(v_line_param[x]))]
        [axs[sub].text(v_line_param[x][j][0] - 100, -.1, v_line_param[x][j][1], color='k', fontsize=16, transform=axs[sub].get_xaxis_transform(), rotation='horizontal') for x in range(len(v_line_param)) for j in range(len(v_line_param[x]))]        

      # Setting the values for all axes.
      #plt.setp(axs, ylim=[-0.5,1], yticks=[-0.5, 0, 0.5, 1]) 
        
      axs[sub].margins(0, 0.05)
      axs[sub].axhline(0, linestyle='--', color='k')
      axs[sub].tick_params(axis='y', which='major', labelsize=16)
      # axs[sub].set_ylim([-0.1, 0.18])
      if plot_legend == True:
        #legend1 = axs[sub].legend(prop={'size': 6}, loc='upper right')
        legend1 = axs[sub].legend(prop={'size': 7}, ncol = 2, bbox_to_anchor=(0.1, 0.6, 0.5, 0.5), fancybox=True, framealpha=0.3)
    
    #show significance
    if test_visual == 'dots':    #draw dots above the plot if statistically significant
        for sub in range(len(subplots)):
          ymin, ymax = axs[sub].get_ylim()
          add_y = 0
          for sub_pred in subplots[sub]:
            p_val = statsmodels.stats.multitest.multipletests(
              prob.T[base_predictors.index(sub_pred)], method='fdr_bh')[1] #correction for multiple comparison with Benjamini-Hochberg correction
            
            marker_size = [50 if p < 0.001 else 30 if p < 0.01 else 10 if p < 0.05 else 0 for p
                           in p_val] #different size of the dot depending on the p value
    
            scatter = axs[sub].scatter((t_scale_whole - time_start)[::1], [ymax + add_y] * len(marker_size), s=marker_size,
                             color=colors[base_predictors.index(sub_pred)], marker='o')
            add_y += (max([max(mean.T[base_predictors.index(sub_pred)]) for sub_pred in subplots[sub]]) - \
                      min([min(mean.T[base_predictors.index(sub_pred)]) for sub_pred in subplots[sub]])) / 7

    for ax in axs:
      ax.label_outer()

    if title != []:
      fig.suptitle(title)
    #fig.text(-0.05, 0.5, 'Regression coefficients', va='center', rotation='vertical')
    fig.subplots_adjust(wspace=0, hspace=0.3, top=0.95, right=0.99, bottom=0.05, left=0.07) 
  return(coef_reg_all, all_predictors, all_pho_scale, mean, sem, prob)

