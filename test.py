import sys
sys.path.append('./util')

import import_beh_data as bi

session = bi.Session('../../../OneDrive - Nexus365/Data/Basal_Ganglia_Cost_Benefit/Data/ThreeChoice_Task/JC06_ThreeChoice_A2a_Drd1/Photometry/Behaviour/A7.5b-2022-11-12-140835.txt')

session.events_and_print

