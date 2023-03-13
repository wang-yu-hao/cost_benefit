import sys, os
import glob
import numpy as np
from parse import *
import operator
from datetime import datetime
import pandas as pd

def import_sessions_photometry_path(dir_folder_session, dir_folder_pho, start_str, sessions_format, photometry_format,
                                    mouse, day, training='all', region=[],
                                    hemisphere=[], exclusion=[]):

  '''
  photometry_format: e.g. for D2 animals ''A1.5f-2019-12-18-090859.ppd' --> 'A{id}_{region}_{hemisphere}-{datetime}.ppd'
  for D1 animals 'D3.3b-2019-12-18-090859.ppd' --> 'D{id}_{region}_{hemisphere}-{datetime}.ppd'
  training='all' = takes all the files from all the sessions
  training='LastFive' = takes only the last 5 sessions in each block (Initial, Reversal1, Reversal2), it uses
  the excel file Block_dates_LastFive which can be found on the server
  '''

  all_sessions_path = glob.glob(os.path.join(dir_folder_session, '{}*'.format(start_str)))
  all_photometry_path = glob.glob(os.path.join(dir_folder_pho, '{}*'.format(start_str)))

  if exclusion != []:
    exclusion_sessions = [os.path.join(dir_folder_session, x) for x in exclusion]
    all_sessions_path = [x for x in all_sessions_path if x not in exclusion_sessions]
    exclusion_photometry = [os.path.join(dir_folder_pho, x) for x in exclusion]
    all_photometry_path = [x for x in all_photometry_path if x not in exclusion_photometry]

  all_sessions_path = [os.path.basename(all_sessions_path[i]) for i in range(len(all_sessions_path))]
  all_photometry_path = [os.path.basename(all_photometry_path[i]) for i in range(len(all_photometry_path))]
  #print(all_sessions_path)
  #print(all_photometry_path)
  list_sessions_m_d = [
    [parse(sessions_format, all_sessions_path[i])['id'],
     datetime.strptime(parse(sessions_format, all_sessions_path[i])['datetime'], '%Y-%m-%d-%H%M%S')] for i in
    range(len(all_sessions_path))]
  #print(list_sessions_m_d)
  sort_sessions_m_d = sorted(list_sessions_m_d, key=operator.itemgetter(0, 1))
  idx_sort_sessions = [list_sessions_m_d.index(x) for x in sort_sessions_m_d]
  all_sessions_path = [all_sessions_path[x] for x in idx_sort_sessions]
  #print(all_sessions_path)
  list_pho_m_d = [
    [parse(photometry_format, all_photometry_path[i])['id'],
     datetime.strptime(parse(photometry_format, all_photometry_path[i])['datetime'], '%Y-%m-%d-%H%M%S')] for i in
    range(len(all_photometry_path))]

  sort_pho_m_d = sorted(list_pho_m_d, key=operator.itemgetter(0, 1))
  idx_sort_pho_m_d = [list_pho_m_d.index(x) for x in sort_pho_m_d]
  all_photometry_path = [all_photometry_path[x] for x in idx_sort_pho_m_d]

  # select animal, day, region, hemisphere to import
  if mouse:
    mouse_id = [parse(photometry_format, all_photometry_path[i])['id'] in mouse
                for i in range(len(all_photometry_path))]
    all_photometry_path = [all_photometry_path[i] for i in np.where(mouse_id)[0]]

    mouse_id = [parse(sessions_format, all_sessions_path[i])['id'] in mouse
                for i in range(len(all_sessions_path))]
    all_sessions_path = [all_sessions_path[i] for i in np.where(mouse_id)[0]]

  if day:
    day_id = [parse(photometry_format, all_photometry_path[i])['datetime'][:-7] in day
              for i in range(len(all_photometry_path))]

    all_photometry_path = [all_photometry_path[i] for i in np.where(day_id)[0]]
    # all_sessions_path = [all_sessions_path[i] for i in np.where(day_id)[0]]
    day_id = [parse(sessions_format, all_sessions_path[i])['datetime'][:-7] in day
              for i in range(len(all_sessions_path))]
    all_sessions_path = [all_sessions_path[i] for i in np.where(day_id)[0]]

  if region:
    region_id = [parse(photometry_format, all_photometry_path[i])['region'] in region
                 for i in range(len(all_photometry_path))]
    all_photometry_path = [all_photometry_path[i] for i in np.where(region_id)[0]]
    all_sessions_path = [all_sessions_path[i] for i in np.where(region_id)[0]]

  if hemisphere:
    hemisphere_id = [parse(photometry_format, all_photometry_path[i])['hemisphere'] in hemisphere
                for i in range(len(all_photometry_path))]
    all_photometry_path = [all_photometry_path[i] for i in np.where(hemisphere_id)[0]]
    all_sessions_path = [all_sessions_path[i] for i in np.where(hemisphere_id)[0]]
  
  if training == 'LastFive':  
      all_photo_tokeep = []
      all_session_tokeep = []
      filename = 'Block_dates_LastFive.xlsx' #Excel file to get the last 5 sessions of each block
      block_names = ['initial', 'reversal1', 'reversal2'] #to loop through the worksheets   

      for photo in all_photometry_path:
        #string = session.split("\\")[-1] #takes only the file name
        animal_id = photo.split("\\")[-1][:5] #get the animal id from the file name
        date = photo.split("\\")[-1][12:-11] #get the date from the file name
        for block in block_names:
            df = pd.read_excel(filename, sheet_name=block) #read the excel file as a dataframe
            ind = (df['Subject_id'] == animal_id) #look for the position of the mouse id in that column
            if str(df.loc[ind, 'Start'].iat[0]).split()[0] <= date <= str(df.loc[ind, 'End'].iat[0]).split()[0]:
                #if the date is included in the date limits of the block, keep the session
                all_photo_tokeep.append(photo)
                all_photometry_path = all_photo_tokeep
                
      for session in all_sessions_path:
        #string = session.split("\\")[-1]
        animal_id = session.split("\\")[-1][:5]
        date = session.split("\\")[-1][6:-11]  
        for block in block_names:
            df = pd.read_excel(filename, sheet_name=block) #read the excel file as a dataframe
            ind = (df['Subject_id'] == animal_id) #look for the position of the mouse id in that column
            if str(df.loc[ind, 'Start'].iat[0]).split()[0] <= date <= str(df.loc[ind, 'End'].iat[0]).split()[0]:
                #if the date is included in the date limits of the block, keep the session
                all_session_tokeep.append(session)
                all_sessions_path = all_session_tokeep
                
  #to have only sessions files matched with photometry files
  names_date = [all_photometry_path[i][:-11] for i in range(len(all_photometry_path))] #take the beginning of file name&date
  #print(all_photometry_path[0])
  names_date = [names_date[i].replace("_DLS", "") for i in range(len(names_date))] 
  names_date = [names_date[i].replace("_DMS", "") for i in range(len(names_date))] #remove DLS and DMS to have the same format as behavioural file name
  names_date = [names_date[i].replace("_R", "") for i in range(len(names_date))] 
  names_date = [names_date[i].replace("_L", "") for i in range(len(names_date))] #remove L and R to have the same format as behavioural file name  
  all_sessions_path = [all_sessions_path[i] for i in range(len(all_sessions_path)) if all_sessions_path[i][:-11] in names_date] #take only sessions if there is a similar photometry file
  print(all_photometry_path)
  print(all_sessions_path)
  #check behavioural and photometry data are correctly paired
  mouse_check = [parse(sessions_format, all_sessions_path[i])['id'] == parse(photometry_format, all_photometry_path[i])['id']
                 for i in range(len(all_photometry_path))]

  day_check = [parse(sessions_format, all_sessions_path[i])['datetime'][:-7] ==
               parse(photometry_format, all_photometry_path[i])['datetime'][:-7]
               for i in range(len(all_photometry_path))]

  if any(mc is False for mc in mouse_check):
    raise ValueError('Mouse is not correctly aligned')
  elif any(dc is False for dc in day_check):
    raise ValueError('Day is not correctly aligned')
  else:
    all_sessions_path = [os.path.join(dir_folder_session, x) for x in
                         [all_sessions_path[i] for i in range(len(all_sessions_path))]]
    all_photometry_path = [os.path.join(dir_folder_pho, x) for x in
                           [all_photometry_path[i] for i in range(len(all_photometry_path))]]
    return all_sessions_path, all_photometry_path
