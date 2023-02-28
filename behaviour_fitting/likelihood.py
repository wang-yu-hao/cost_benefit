'''
Negative log-likelihoods (to be minimised)
'''

import numpy as np
import scipy.stats as stats

# Rescorla-Wagner
def rw(sessions, Q_0, alpha, e):
    
    '''
    sessions: list of experiment sessions of a single subject
    Q_0: initial Q values
    alpha: learning rate
    e: noise term (similar to softmax temperature)
    '''

    L = 0 # Negative log likelihood, initialised at 0
    
    # Storage for latent variables and choice probabilities:
    Q_table = []
    p_table = []

    for sesh in sessions: # Iterate over sessions

        Q_list = [] # Temp. storage
        p_list = []
        Q = {'1.0': Q_0, '0.5': Q_0, '0.2': Q_0} # Initial Q values before each session

        td = sesh.trial_data

        for t in range(len(td['choices'])): # Iterate over trials in a session

            if td['free_choice'][t] == False:

                p = 1

            else:
                
                p = stats.norm.cdf((2 * Q[str(td['proba_choosed'][t])] - Q[str(td['prob_high'][t])] - Q[str(td['prob_low'][t])])/(2**0.5 * e))

                # p = np.exp(beta * Q[str(td['proba_choosed'][t])]) / (np.exp(beta * Q[str(td['prob_high'][t])]) + np.exp(beta * Q[str(td['prob_low'][t])])) # Model predicted probability of the action taken

            delta = td['outcomes'][t] - Q[str(td['proba_choosed'][t])] # RPE
            Q[str(td['proba_choosed'][t])] += alpha * delta # R-W belief update

            Q_list.append(list(Q.values()))
            p_list.append(p)

            L += -np.log(p)

        Q_table.append(Q_list)
        p_table.append(p_list)

    # Q_table = np.array(Q_table)
    # p_table = np.array(p_table)


    return [L, Q_table, p_table]