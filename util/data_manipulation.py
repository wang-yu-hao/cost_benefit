import util.import_beh_data as di

# session = di.Session('../../OneDrive - Nexus365/Data/Basal_Ganglia_Cost_Benefit/Data/ThreeChoice_Task/JC06_ThreeChoice_A2a_Drd1/Photometry/Behaviour/A7.5b-2022-11-12-140835.txt')

# session.events_and_print

def load_all_sessions(cohort):
    
    experiment = di.Experiment('../../OneDrive - Nexus365/Data/Basal_Ganglia_Cost_Benefit/Data/ThreeChoice_Task/JC0{}_ThreeChoice_A2a_Drd1/Photometry/Behaviour'.format(cohort))

    return experiment