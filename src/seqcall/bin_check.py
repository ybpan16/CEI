import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

def winsorization_median(x, trim=0.05):
    """
    Winsorize the data by replacing the lower and upper trim percentiles with the median.
    """
    x = np.asarray(x, float)
    if len(x) == 0:
        return x
    
    lower_bound, upper_bound = np.quantile(x, [trim, 1-trim])
    x_updated = np.clip(x, lower_bound, upper_bound)
    x_median = np.median(x_updated)
    
    return x_median

def z_proto(
        xB, xP,
        n_bins=20, nmin_merge=200,
        trim=0.05
):
    xB = np.asarray(xB, float)
    xP = np.asarray(xP, float)
    if xB.shape != xP.shape:
        raise ValueError("xB and xP must have the same length")

    SB = np.sum(xB)
    SP = np.sum(xP)
    
    # Calculate the pooled proportion
    p_pool = (xB + xP) / (SB + SP)
    pB = xB / SB
    pP = xP / SP
    d = pB - pP

    # Z2_proto
    Var0 = p_pool * (1-p_pool) * (1/SB + 1/SP)
    Z2_proto = d**2 / Var0

    # Split the Z2_proto into bins
    p_logit = np.log(p_pool / (1-p_pool))
    bins = np.linspace(np.min(p_logit), np.max(p_logit), n_bins)
    hist, bin_edges = np.histogram(p_logit, bins=bins, weights=Z2_proto)
    id_bins = np.digitize(p_logit, bin_edges) - 1
    id_bins = np.clip(id_bins, 0, bin_edges.size - 2)

    # Merge bins until each bin has at least nmin_merge entries
    merged_bins = []
    start = len(bin_edges)-1
    while start > 1:
        tmp = start
        idx = (id_bins == tmp)
        n_b = int(np.sum(idx))
        print(n_b)
        while (n_b < nmin_merge) and (tmp > 1):
            tmp -= 1
            idx |= (id_bins == tmp)
            n_b = int(np.sum(idx))
            bin_edges = np.delete(bin_edges, tmp + 1)
        start = tmp - 1

    id_bins = np.digitize(p_logit, bin_edges) - 1
    id_bins = np.clip(id_bins, 0, bin_edges.size - 2)

    # Summarize each bin
    med_Z2_proto = []
    centers = []
    n_bs = []
    id_unique = np.unique(id_bins)
    print(id_unique)
    for i in id_unique:
        n_bs.append(np.sum(id_bins == i))
        z_proto = Z2_proto[id_bins == i]
        med = winsorization_median(z_proto)
        med_Z2_proto.append(med)
        centers.append(0.5 * (bin_edges[i] + bin_edges[i+1]))

    plt.plot(centers, med_Z2_proto, marker='o')
    plt.show()

    return n_bs
