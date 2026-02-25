import numpy as np

def find_chunks(array):
    import numpy as np
#    array[np.isnan(array)] = 0
    starts = np.diff(array) == 1
    stops = np.diff(array) == -1
    starts_ix = np.nonzero(starts)[0] + 1
    stops_ix = np.nonzero(stops)[0] + 1
    if len(starts_ix) != len(stops_ix): stops_ix = np.append(stops_ix, len(array))
    return starts_ix, stops_ix

def slice_tuple(slice_):
    return slice_.start, slice_.stop, slice_.step

def run_permstats(diff):
    import mne
    import numpy as np
    t_obs, clusters, clusters_pv, H0 = mne.stats.permutation_cluster_1samp_test(diff)
    sta = []
    sto = []
    for c in zip(clusters):
        sta.append(c[0][0][0])
        sto.append(c[0][0][len(c[0][0])-1])
    
    return np.array(sta), np.array(sto), np.array(clusters_pv)

### OLD V FROM OTHER MAC
#def run_permstats(diff):
#    import mne
#    import numpy as np
#    t_obs, clusters, clusters_pv, H0 = mne.stats.permutation_cluster_1samp_test(diff)
#    sta = []
#    sto = []
#    p = []
#    for c, p_val in zip(clusters, clusters_pv):
#        sta.append(slice_tuple(c[0])[0])
#        sto.append(slice_tuple(c[0])[1])
#        p.append(p_val)
#    return np.array(sta), np.array(sto), np.array(p)