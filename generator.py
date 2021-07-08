import numpy as np
import scipy.stats as sps
from help_functions import deltas_states

def sample(n_timepoints, n_states, state_std=1):
    # n = number of timepoints
    # p = number of states
    # r = variability of state lengths
    x_0 = np.linspace(0, n_timepoints, n_states + 1).astype(int)[1: -1]
    x_r = []
    q = state_std * (n_timepoints / n_states) - 0.5

    for i in range(n_states - 1):
        while True:
            x = x_0[i] + np.random.randint(-q, q + 1)
            if (i > 0 and x_r[i - 1] >= x) or (i == 0 and x < 1) \
                or (x > n_timepoints - (n_states-i)-1):
                continue
            else:
                break

        x_r.append(x)

    x_r = np.array(x_r)
    # x_d = np.concatenate(([x_r[0]], x_r[1:] - x_r[:-1], [n - x_r[-1]]))
    bounds = np.zeros(n_timepoints).astype(int)
    bounds[x_r] = 1
    states = deltas_states((bounds))
    
    print(max(states))
    # x_d = np.concatenate(([x_r[0]], x_r[1:] - x_r[:-1], [n - x_r[-1]]))

    return bounds, states


def generate_simulated_data_HRF(ntime=200, 
                                nvox=50, 
                                nstates=15,
                                nsub=1, 
                                group_std=0, 
                                sub_std=0.1, 
                                sub_evprob=0., 
                                length_std=1, 
                                peak_delay=6, 
                                peak_disp=1, 
                                extime=2, 
                                TR=2.47, 
                                TRfactor=1,  
                                rep=500):

    np.random.seed(rep)
    state_pattern = np.random.randn(nvox, nstates)

    bounds, state_labels = sample(ntime, nstates, length_std)
    nbound_id = np.array(np.where(bounds == 0)[0])
    bound_id = np.array(np.where(bounds == 1)[0])
    
    spmhrf = spm_hrf_compat(np.arange(0, 30, TR), peak_delay=peak_delay, peak_disp=peak_disp)
    subject_data = np.zeros([nsub,  ntime, nvox])

    group_noise = group_std * np.random.randn(ntime + extime, nvox)
    bold_group_noise = convolve_with_hrf(group_noise, spmhrf, ntime, extime)
    
    subj_bounds = np.zeros([nsub, ntime])
    
    for s in range(nsub):

        p_bound = sub_evprob
        p_notbound = sub_evprob / (len(nbound_id) / (nstates - 1))
        
        sampl_bound = np.random.binomial(1, p_bound, nstates - 1)
        sampl_nbound = np.random.binomial(1, p_notbound, len(nbound_id))

        rem_bound = bound_id[np.nonzero(sampl_bound)[0]]
        add_bound = nbound_id[np.nonzero(sampl_nbound)[0]]

        new_bounds = np.concatenate((rem_bound, add_bound))

        sub_state_labels = np.copy(state_labels)

        for t in new_bounds:
            
            # a boundary disappears
            if bounds[t] == 1:
                ev = np.argwhere(sub_state_labels == sub_state_labels[t])
                sub_state_labels[ev] = sub_state_labels[t-1]

            # a state boundary appears
            elif bounds[t] == 0:
                times = np.arange(t, ntime, 1)
                ev = np.argwhere(sub_state_labels[times] == sub_state_labels[t])
                sub_state_labels[times[ev]] = np.amax(sub_state_labels) + 1       
        
        if np.amax(sub_state_labels) > (nstates-1):
            addstate_pattern = np.random.randn(nvox, np.amax(sub_state_labels)-(nstates-1))
        
        subj_bounds[s, 1:] = np.diff(sub_state_labels)
        
        subject_event_data = np.zeros([ntime + extime, nvox])
        for t in range(0, ntime):
            if sub_state_labels[t] < nstates:
                subject_event_data[t, :] = state_pattern[:, sub_state_labels[t]]
            else:
                subject_event_data[t, :] = addstate_pattern[:, sub_state_labels[t]-nstates]


        for te in range(ntime, ntime + extime):
            subject_event_data[te, :] = subject_event_data[ntime - 1, :]

        bold_subj_data = convolve_with_hrf(subject_event_data, spmhrf, ntime, extime)

        subj_noise = sub_std * np.random.randn(ntime + extime, nvox)
        bold_subj_noise = convolve_with_hrf(subj_noise, spmhrf, ntime, extime)

        subject_data[s, :, :] = bold_subj_data + bold_subj_noise + bold_group_noise

    return bounds, subject_data, subj_bounds



def spm_hrf_compat(t,
                   peak_delay=6,
                   under_delay=16,
                   peak_disp=1,
                   under_disp=1,
                   p_u_ratio=6,
                   normalize=True):

    if len([v for v in [peak_delay, peak_disp, under_delay, under_disp]
            if v <= 0]):
        raise ValueError("delays and dispersions must be > 0")
    # gamma.pdf only defined for t > 0
    hrf = np.zeros(t.shape, dtype=np.float)
    pos_t = t[t > 0]
    peak = sps.gamma.pdf(pos_t,
                         peak_delay / peak_disp,
                         loc=0,
                         scale=peak_disp)
    undershoot = sps.gamma.pdf(pos_t,
                               under_delay / under_disp,
                               loc=0,
                               scale=under_disp)
    hrf[t > 0] = peak - undershoot / p_u_ratio
    if not normalize:
        return hrf
    return hrf / np.max(hrf)


def convolve_with_hrf(signal, hrf, ntime, extime):

    nvox = signal.shape[1]

    bold = np.zeros([ntime + extime + len(hrf) - 1, nvox])
    
    for n in range(nvox):
        bold[:, n] = np.convolve(signal[:, n], hrf)
    bold = bold[extime:extime+ntime, :]
    
    return bold
