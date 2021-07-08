import numpy as np
from sklearn.metrics import confusion_matrix
from scipy.spatial.distance import cdist
from scipy.stats import  ttest_ind, zscore
from scipy.optimize import linear_sum_assignment
from gsbs_extra import GSBS
#from brainiak.eventseg.event import EventSegment as HMM
from joblib import Parallel, delayed

def deltas_states(deltas: np.ndarray) -> np.ndarray:
    deltas.astype(int)
    states = np.zeros(deltas.shape[0], int)
    for i, delta in enumerate(deltas[1:]):
        states[i + 1] = states[i] + 1 if delta else states[i]

    return states

def fit_metrics_simulation(real_bounds, recovered_bounds):
    
    real_bounds[real_bounds>0] = 1
    real_bounds.astype(int)
    real_states = deltas_states(real_bounds)
    real_locations = np.where(real_bounds)[0]

    recovered_bounds[recovered_bounds>0] = 1
    recovered_bounds.astype(int)
    recovered_states = deltas_states(recovered_bounds)    
    recovered_locations = np.where(recovered_bounds)[0]
    
    dist = compute_prediction_distance(real_bounds, recovered_bounds)

    simm, simz = correct_fit_metric(real_states, recovered_states)

    return simm, simz, dist

def compute_prediction_distance(y_true, y_pred):

    real_states = deltas_states(y_true)
    real_locations = np.where(y_true)[0]

    recovered_states = deltas_states(y_pred)    
    recovered_locations = np.where(y_pred)[0]    

    dist = np.zeros(np.max(recovered_states)+1)

    for i, pred_loc in enumerate(recovered_locations):
        loc_difference = np.abs(real_locations - pred_loc)
        dist[i] = np.min(loc_difference)
        loc = np.argmin(loc_difference)
        if pred_loc < real_locations[loc]:
            dist[i] = -dist[i]

    return dist



#function to state detection with HMM and compute the relevant fit metrics
def compute_fits_hmm(data:np.ndarray, k:int, mindist:int, type='HMM', y=None, t1=None, ind1=None, zs=False):
    """
    if type == 'HMM':
        hmm = HMM(k)
    elif type == 'HMMsplit':
        hmm = HMM(k, split_merge=True)
    """
    hmm = "HMM"
    if zs == True:
        data = zscore(data, axis=0, ddof=1)
    hmm.fit(data)

    if y is None:
        tdata = data
    else:
        if zs == True:
            y = zscore(y, axis=0, ddof=1)
        tdata=y

    _, LL_HMM = hmm.find_events(tdata)

    hmm_bounds = np.insert(np.diff(np.argmax(hmm.segments_[0], axis=1)), 0, 0).astype(int)

    if t1 is None and ind1 is None:
        ind = np.triu(np.ones(tdata.shape[0], bool), mindist)
        z = GSBS._zscore(tdata)
        t = np.cov(z)[ind]
    else:
        ind = ind1
        t = t1

    stateseq = deltas_states(deltas=hmm_bounds)[:, None]
    diff, same, alldiff = (lambda c: (c == 1, c == 0, c > 0))(cdist(stateseq, stateseq, "cityblock")[ind])
    WAC_HMM = np.mean(t[same]) - np.mean(t[alldiff])
    tdist_HMM = 0 if sum(same) < 2 else ttest_ind(t[same], t[diff], equal_var=False)[0]

    return LL_HMM, WAC_HMM, tdist_HMM, hmm_bounds, t, ind

# subfunction to compute reliability
def compute_reliability(data, n_jobs=-1):
    
    with Parallel(n_jobs=n_jobs) as p:
        reliability_sim, reliability_simz = \
            zip(*p(delayed(compute_reliability_parallel(data, i)) for i in range(data.shape[0])))

    return reliability_sim, reliability_simz


def compute_reliability_parallel(data, subj):
    indlist = np.arange(0, data.shape[0])
    state = deltas_states(data[subj, :].astype(int))
    
    other_subjs = np.setdiff1d(indlist, subj)
    avgdata = np.mean(data[other_subjs, :], axis=0)

    #get the k most observed boundaries and compute accuracy on group level with fixed k
    k = np.int(np.max(state))
    group_deltas_loc = np.argsort(-avgdata)[0:k]
    group_deltas = np.zeros(avgdata.shape)
    group_deltas[group_deltas_loc] = 1

    states_group = deltas_states(group_deltas.astype(int))
    reliability_sim, reliability_simz = correct_fit_metric(states_group, state)

    return reliability_sim, reliability_simz




def compute_reliability_pcor(data):
    indlist = np.arange(0, data.shape[0])
    reliability_pcor = np.zeros(len(indlist))

    for i in indlist:
        #correlate each subject with the rest of the group
        avgdata = np.mean(data[np.setdiff1d(indlist, i), :], axis=0)
        reliability_pcor[i] = np.corrcoef(avgdata, data[i,:])[0,1]

    return reliability_pcor


def correct_fit_metric(y_true, y_pred, n_jobs=-1, n_perm=1000):

    true_accuracy = get_accuracy(y_true, y_pred)
    
    with Parallel(n_jobs=n_jobs) as p:
        permutations = p(delayed(randomize_fit)(y_true, y_pred) \
            for i in range(n_perm))
    
    permutations = np.array(permutations)
    avg_perm = np.mean(permutations)

    permutations_zscored = (true_accuracy - avg_perm) / np.std(permutations)   
    permutations_average = (true_accuracy - avg_perm) / (1 - avg_perm)

    return permutations_average, permutations_zscored


def randomize_fit(y_true, y_pred):

    nclass_true = len(np.unique(y_true))
    nclass_pred = len(np.unique(y_pred))
    nitems = len(y_true)
    
    states = []

    for label in [nclass_true, nclass_pred]:

        bound_loc = np.random.choice(nitems, [label - 1, 1], replace=False)
        bounds = np.zeros((nitems, 1)).astype(int)
        bounds[bound_loc] = 1
        states.append(deltas_states(bounds))
        
    simulated_accuracy = get_accuracy(states[0], states[1])

    return simulated_accuracy


def get_accuracy(y_true, y_pred):
    
    c = confusion_matrix(y_true, y_pred)
    row_ind, col_ind = linear_sum_assignment(-c)
    accuracy = c[row_ind, col_ind].sum() / len(y_true)

    return accuracy

