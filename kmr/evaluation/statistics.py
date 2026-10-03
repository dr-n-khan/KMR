import numpy as np
from scipy.stats import ttest_rel, wilcoxon, t

def paired_stats(kmr,baseline,alpha=.05):
    a=np.asarray(kmr,float); b=np.asarray(baseline,float); d=a-b; n=len(d); mean=d.mean(); sd=d.std(ddof=1) if n>1 else 0.; se=sd/np.sqrt(n) if n else np.nan; crit=t.ppf(1-alpha/2,n-1) if n>1 else np.nan
    tp=float(ttest_rel(a,b).pvalue) if n>1 else np.nan
    try: wp=float(wilcoxon(d).pvalue)
    except ValueError: wp=1.0
    return {'mean_difference':float(mean),'ci95_low':float(mean-crit*se),'ci95_high':float(mean+crit*se),'cohens_d_paired':float(mean/sd) if sd else 0.,'paired_t_p':tp,'wilcoxon_p':wp}
