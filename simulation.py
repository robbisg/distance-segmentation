import numpy as np
from statesegmentation import GSBS
import gsbs_extra
from typing import Tuple
from scipy.spatial.distance import cdist
from scipy.stats import pearsonr, ttest_ind, mannwhitneyu
from sklearn.model_selection import KFold
from hrf_estimation import hrf
import timeit
from help_functions import fit_metrics_simulation, compute_fits_hmm, deltas_states
from brainiak.eventseg.event import EventSegment as HMM
from importlib import reload


def basic_simulation():

    results = dict()
    
    real_bounds, subData, _ = generate_simulated_data_HRF(length_std=l, nstates=n, TRfactor=TRfactor, rep=rep)

    states = gsbs_extra.GSBS(kmax=n, x=subData[0, :, :], finetune=finetune)
    states.fit()

    recovered_bounds = np.double(states.get_bounds(k=n)>0)
    permut_accuracy, permut_zaccuracy, distance = fit_metrics_simulation(real_bounds, recovered_bounds)

    results['perm_accuracy'] = permut_accuracy
    results['perm_zaccuracy'] = permut_zaccurcy
    results['distance'] = distance
    results['real_bounds'] = states.bounds
    results['real_states'] = deltas_states(real_bounds)


# simulation 1, vary state length and estimate how accurately we can recover state boundaries
def run_simulation_evlength(length_std, nstates_list, run_HMM, rep, TRfactor=1, finetune=1):

    res = dict()
    list2 = ['dists_GS','dists_HMM', 'dists_HMMsplit']

    nstd = len(length_std)
    nstates = len(nstates_list)


    for key in list2:
        res[key] = np.zeros([nstd, nstates, nstates_list[-1]])

    list = ['sim_GS', 'sim_HMM','sim_HMMsplit', 'simz_GS', 'simz_HMM', 'simz_HMMsplit']
    for key in list:
        res[key] = np.zeros([nstd, nstates])
    res['statesreal'] = np.zeros([nstd, nstates, ntime])
    res['bounds'] = np.zeros([nstd, nstates, ntime])
    res['bounds_HMMsplit'] = np.zeros([nstd, nstates, ntime])

    for idxl, l in enumerate(length_std):
        for idxn, n in enumerate(nstates_list):
            print(rep, l)
            
            bounds, subData, _ = generate_simulated_data_HRF(length_std=l, nstates=n, TRfactor=TRfactor, rep=rep)
            
            states = gsbs_extra.GSBS(kmax=n, x=subData[0,:,:], finetune=finetune)
            states.fit()
            
            
            
            
            res['sim_GS'][idxl,idxn], res['simz_GS'][idxl, idxn],res['dists_GS'][idxl,idxn,0:n] = fit_metrics_simulation(bounds, np.double(states.get_bounds(k=n)>0))
            res['bounds'][idxl,idxn,:] = states.bounds
            res['statesreal'][idxl,idxn,:] = deltas_states(bounds)

            if run_HMM is True:
                ev = HMM(n, split_merge=False)
                ev.fit(subData[0, :, :])
                hmm_bounds = np.insert(np.diff(np.argmax(ev.segments_[0], axis=1)), 0, 0).astype(int)
                res['sim_HMM'][idxl, idxn], res['simz_HMM'][idxl, idxn], res['dists_HMM'][idxl, idxn, 0:n] = fit_metrics_simulation(bounds, hmm_bounds)

                ev = HMM(n, split_merge=True)
                ev.fit(subData[0, :, :])
                hmm_bounds_split = np.insert(np.diff(np.argmax(ev.segments_[0], axis=1)), 0, 0).astype(int)
                res['sim_HMMsplit'][idxl, idxn],  res['simz_HMMsplit'][idxl, idxn], res['dists_HMMsplit'][idxl, idxn, 0:n] = fit_metrics_simulation(bounds, hmm_bounds_split)
                res['bounds_HMMsplit'][idxl, idxn, :] = hmm_bounds_split

    return res


#simulation 2, how do the different fit measures compare, depending on how many states there are (more states should cause more similarity between distinct states)
def run_simulation_compare_nstates( nstates_list, mindist, run_HMM, finetune, zs, rep):

    res2 = dict()

    metrics = ['LL', 'WAC', 'tdist', 'mdist', 'meddist', 'mwu']
    methods = ['HMM', 'GS']
    types = ['optimum']

    for metric, method, type_ in itertools.product(metrics, methods, types):
        key = type_+"_"+metric+"_"+method
        res2[key] = np.zeros(len(nstates_list))


    metrics = ['LL', 'WAC', 'tdist']
    methods = ['GS', 'HMM', 'HMMsplit']
    types = ['sim', 'simz']

    for metric, method, type_ in itertools.product(metrics, methods, types):
        key = type_+"_"+metric+"_"+method
        res2[key] = np.zeros(len(nstates_list))

    

    list2 = ['tdist', 'WAC', 'mdist', 'meddist',  'LL_HMM', 'WAC_HMM', 
             'tdist_HMM', 'fit_W_mean', 'fit_W_std', 'fit_Ball_mean', 
             'fit_Ball_std', 'fit_Bcon_mean', 'fit_Bcon_std']
    for i in list2:
        res2[i] = np.zeros([len(nstates_list), maxK+1])

    for idxl, l in enumerate(nstates_list):
        print(rep, l)
        real_bounds, subData,_ = generate_simulated_data_HRF(nstates=l, rep=rep)

        data = subData[0, :, :]

        states = gsbs_extra.GSBS(x=data, kmax=maxK, outextra=True, dmin=mindist, finetune=finetune)
        states.fit()
        res2['sim_GS_tdist'][idxl],  res2['simz_GS_tdist'][idxl], dist = \
                fit_metrics_simulation(real_bounds, states.deltas)
        res2['sim_GS_WAC'][idxl], res2['simz_GS_WAC'][idxl], dist = \
                fit_metrics_simulation(real_bounds, states.get_deltas(k=states.optimum_WAC))

        for metric in ['tdist', 'WAC', 'meddist', 'mdist']:

            res2[metric][idxl, :] = getattr(states, metric)

            attr = "optimum_" + metric
            optimum = getattr(states, attr)

            key = "optimum_" + metric + "_GS"
            res2[key][idxl] = optimum


        res2['fit_W_mean'][idxl, :] = states.all_m_W
        res2['fit_W_std'][idxl, :] = states.all_sd_W
        res2['fit_Ball_mean'][idxl, :] = states.all_m_Ball
        res2['fit_Ball_std'][idxl, :] = states.all_sd_Ball
        res2['fit_Bcon_mean'][idxl, :] = states.all_m_Bcon
        res2['fit_Bcon_std'][idxl, :] = states.all_sd_Bcon

        # HMM
        t = None
        ind = None

        for i in range(2, maxK):
            res2['LL_HMM'][idxl, i], res2['WAC_HMM'][idxl, i], res2['tdist_HMM'][idxl, i], \
            hmm_bounds, t, ind = compute_fits_hmm(data, i, mindist, type='HMM', y=None, t1=t, ind1=ind, zs=zs)

        for metric in ['LL', 'WAC', 'tdist']:
            keyword = "%s_HMM" % (metric)
            optimum_ncluster = np.argmax(res2[keyword][idxl])
    
            if metric == 'LL':
                optimum_ncluster = np.argmax(res2[keyword][idxl][2:90]) + 2

            res2['optimum_%s' % keyword] = optimum_ncluster

            for method in ['HMM', 'HMMsplit']:

                _, _, _, recovered_bounds, t, ind = compute_fits_hmm(
                                                        data=data, 
                                                        k=optimum_ncluster, 
                                                        mindist=1, 
                                                        type=method, 
                                                        y=None, 
                                                        t1=t, 
                                                        ind1=ind)
                
                simm, simz, dist = fit_metrics_simulation(real_bounds, recovered_bounds)

                keyword = 'sim_%s_%s' % (metric, method)
                res2[keyword][idxl] = simm

                keyword = 'simz_%s_%s' % (metric, method)
                res2[keyword][idxl] = simz

    return res2


#simulation 3, can we correctly estimate the number of states in the group when there is ideosyncracy in state boundaries between participants?
def run_simulation_sub_noise( CV_list, sub_std_list, kfold_list, nsub, rep):

    res3=dict()
    list=['optimum', 'sim_GS','sim_GS_fixK', 'simz_GS', 'simz_GS_fixK']
    for key in list:
        res3[key] = np.zeros([np.shape(CV_list)[0], np.shape(sub_std_list)[0], np.shape(kfold_list)[0]])
    res3['tdist'] = np.zeros([np.shape(CV_list)[0], np.shape(sub_std_list)[0], np.shape(kfold_list)[0], maxK + 1])

    list = ['optimum_subopt', 'sim_GS_subopt', 'simz_GS_subopt']
    for key in list:
        res3[key] = np.zeros([np.shape(sub_std_list)[0], nsub])

    for idxs, s in enumerate(sub_std_list):
        bounds, subData,_ = generate_simulated_data_HRF(sub_std=s, nsub=nsub, rep=rep)

        for idxi, i in enumerate(kfold_list):
            print(rep, s, i)
            if i>1:
                kf = KFold(n_splits=i, shuffle=True)
                for idxl, l in enumerate(CV_list):

                    tdist_temp = np.zeros([i,maxK+1]);  optimum_temp = np.zeros(i); GS_sim_temp = np.zeros(i)
                    GS_sim_temp_fixK = np.zeros(i); simz_temp = np.zeros(i); simz_temp_fixK = np.zeros(i)

                    count=-1
                    for train_index, test_index in kf.split(np.arange(0,np.max(kfold_list))):
                        count=count+1
                        print(count)
                        if l is False:
                            states = gsbs_extra.GSBS(x=np.mean(subData[test_index, :, :], axis=0), kmax=maxK)
                        elif l is True:
                            states = gsbs_extra.GSBS(x=np.mean(subData[train_index, :, :], axis=0), y=np.mean(subData[test_index, :, :], axis=0), kmax=maxK)
                        states.fit()

                        optimum_temp[count] = states.nstates
                        tdist_temp[count, :] = states.tdists
                        GS_sim_temp[count], simz_temp[count], dist = fit_metrics_simulation(bounds, states.deltas)
                        GS_sim_temp_fixK[count] , simz_temp_fixK[count], dist = fit_metrics_simulation(bounds, states.get_deltas(k=nstates))

                    res3['optimum'][idxl, idxs, idxi] = np.mean(optimum_temp)
                    res3['sim_GS'][idxl, idxs, idxi] = np.mean(GS_sim_temp)
                    res3['sim_GS_fixK'][idxl, idxs, idxi] = np.mean(GS_sim_temp_fixK)
                    res3['simz_GS'][idxl, idxs, idxi] = np.mean(simz_temp)
                    res3['simz_GS_fixK'][idxl, idxs, idxi] = np.mean(simz_temp_fixK)
                    res3['tdist'][idxl, idxs, idxi, :] = tdist_temp.mean(0)

            else:
                states = gsbs_extra.GSBS(x=np.mean(subData[:, :, :], axis=0), kmax=maxK)
                states.fit()

                res3['optimum'][:, idxs, idxi] = states.nstates
                res3['sim_GS'][:, idxs, idxi], res3['simz_GS'][:, idxs, idxi],dists = fit_metrics_simulation(bounds, states.deltas)
                res3['sim_GS_fixK'][:, idxs, idxi],res3['simz_GS_fixK'][:, idxs, idxi],dists = fit_metrics_simulation(bounds, states.get_deltas(k=nstates))
                res3['tdist'][:, idxs, idxi, :] = states.tdists

                # subbounds = states.fitsubject(subData)
                # for isub in range(nsub):
                #     res3['optimum_subopt'][idxs, isub] = np.shape(subbounds[isub][subbounds[isub]>0])[0]
                #     res3['sim_GS_subopt'][idxs, isub], res3['simz_GS_subopt'][idxs, isub], dists = fit_metrics_simulation(bounds, subbounds[isub])
    return res3


#simulation 4, can we correctly estimate the number of states in the group when there is ideosyncracy in state boundaries between participants?
def run_simulation_sub_specific_states( CV_list, sub_evprob_list, kfold_list, sub_std, nsub, rep):

    res4=dict()
    list=['optimum', 'sim_GS','sim_GS_fixK',  'simz_GS', 'simz_GS_fixK']
    for key in list:
        res4[key] = np.zeros([np.shape(CV_list)[0], np.shape(sub_evprob_list)[0], np.shape(kfold_list)[0]])
    res4['tdist'] = np.zeros([np.shape(CV_list)[0], np.shape(sub_evprob_list)[0], np.shape(kfold_list)[0], maxK + 1])
    list = ['optimum_subopt', 'sim_GS_subopt', 'simz_GS_subopt']
    for key in list:
        res4[key] = np.zeros([np.shape(sub_evprob_list)[0], nsub])

    for idxs, s in enumerate(sub_evprob_list):
        bounds, subData, subbounds = generate_simulated_data_HRF(sub_evprob=s, nsub=nsub, sub_std=sub_std, rep=rep)

        for idxi, i in enumerate(kfold_list):
            print(rep, s, i)
            if i > 1:
                kf = KFold(n_splits=i, shuffle=True)

                for idxl, l in enumerate(CV_list):

                    tdist_temp = np.zeros([i,maxK+1]);  optimum_temp = np.zeros(i); GS_sim_temp = np.zeros(i)
                    GS_sim_temp_fixK = np.zeros(i); simz_temp = np.zeros(i); simz_temp_fixK = np.zeros(i)

                    count = -1
                    for train_index, test_index in kf.split(np.arange(0, np.max(kfold_list))):
                        count = count + 1
                        if l is False:
                            states = gsbs_extra.GSBS(x=np.mean(subData[test_index, :, :], axis=0), kmax=maxK)
                        elif l is True:
                            states = gsbs_extra.GSBS(x=np.mean(subData[train_index, :, :], axis=0),
                                            y=np.mean(subData[test_index, :, :], axis=0), kmax=maxK)
                        states.fit()

                        optimum_temp[count] = states.nstates
                        tdist_temp[count, :] = states.tdists
                        GS_sim_temp[count],  simz_temp[count], dist = fit_metrics_simulation(
                            bounds, states.bounds)
                        GS_sim_temp_fixK[count], simz_temp_fixK[
                            count], dist = fit_metrics_simulation(bounds, states.get_bounds(k=nstates))

                    res4['optimum'][idxl, idxs, idxi] = np.mean(optimum_temp)
                    res4['sim_GS'][idxl, idxs, idxi] = np.mean(GS_sim_temp)
                    res4['sim_GS_fixK'][idxl, idxs, idxi] = np.mean(GS_sim_temp_fixK)
                    res4['simz_GS'][idxl, idxs, idxi] = np.mean(simz_temp)
                    res4['simz_GS_fixK'][idxl, idxs, idxi] = np.mean(simz_temp_fixK)
                    res4['tdist'][idxl, idxs, idxi, :] = tdist_temp.mean(0)

            else:
                states = gsbs_extra.GSBS(x=np.mean(subData[:, :, :], axis=0), kmax=maxK)
                states.fit()

                res4['optimum'][:, idxs, idxi] = states.nstates
                res4['sim_GS'][:, idxs, idxi], res4['simz_GS'][:, idxs, idxi],dists = fit_metrics_simulation(bounds, states.bounds)
                res4['sim_GS_fixK'][:, idxs, idxi], res4['simz_GS_fixK'][:, idxs, idxi],dists = fit_metrics_simulation(bounds, states.get_bounds(k=nstates))
                res4['tdist'][:, idxs, idxi, :] = states.tdists

                # subbounds = states.fitsubject(subData)
                # for isub in range(nsub):
                #     res4['optimum_subopt'][idxs, isub] = np.max(subbounds[isub])
                #     res4['sim_GS_subopt'][idxs, isub], res4['simz_GS_subopt'][idxs, isub], dists = fit_metrics_simulation(bounds,subbounds[isub])

    return res4

# simulation 5, vary the peak and dispersion of the HRF
def run_simulation_hrf_shape( nstates_list, peak_delay_list, peak_disp_list, rep):
    print(rep)
    res5=dict()
    list=['optimum']
    for key in list:
        res5[key] = np.zeros([np.shape(nstates_list)[0], np.shape(peak_delay_list)[0], np.shape(peak_disp_list)[0]])

    for idxe,e in enumerate(nstates_list):
        for idxde, de in enumerate(peak_delay_list):
            for idxdp, dp in enumerate(peak_disp_list):

                bounds, subData,_ = generate_simulated_data_HRF(nstates=e, peak_delay=de, peak_disp=dp, rep=rep)
                states = gsbs_extra.GSBS(x=subData[0,:,:],kmax=maxK)
                states.fit()
                res5['optimum'][idxe, idxde, idxdp] = states.nstates

    return res5


def run_simulation_computation_time( nstates, rep):
    bounds, subData,_ = generate_simulated_data_HRF(rep=rep)
    res6 = dict()
    res6['duration_GSBS'] = np.zeros([nstates])
    res6['duration_HMM_fixK'] = np.zeros([nstates])
    res6['duration_HMMsm_fixK'] = np.zeros([nstates])

    for i in range(2,nstates):
        print(rep, i)
        states = gsbs_extra.GSBS(x=subData[0, :, :], kmax=i)
        tic = timeit.default_timer()
        states.fit()
        res6['duration_GSBS'][i] = timeit.default_timer()-tic

        tic = timeit.default_timer()
        ev = HMM(i, split_merge=False)
        ev.fit(subData[0, :, :])
        res6['duration_HMM_fixK'][i] = timeit.default_timer() - tic

        tic = timeit.default_timer()
        ev = HMM(i, split_merge=True)
        ev.fit(subData[0, :, :])
        res6['duration_HMMsm_fixK'][i] = timeit.default_timer() - tic

    res6['duration_HMM_estK'] = np.cumsum(res6['duration_HMM_fixK'])
    res6['duration_HMMsm_estK'] = np.cumsum(res6['duration_HMMsm_fixK'])

    return res6