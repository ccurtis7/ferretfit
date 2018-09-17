import sys
import os
to_add = os.getcwd().split('/scripts')[0]
if not to_add in sys.path:
    sys.path.insert(0, to_add)

import diff_classifier.aws as aws
import pandas as pd
import boto3
import numpy as np
import statsmodels.api as sm
from scipy.optimize import leastsq
from scipy.stats import sem
import ferretfit.ferretfit as ff

folder = 'ferret_tracking/09_12_18_all_data'
s3 = boto3.resource('s3')
my_bucket = s3.Bucket('ccurtis.data')

all_files = []
for object_summary in my_bucket.objects.filter(Prefix=folder):
    all_files.append(object_summary.key)
    
all_csvs = []
for num in all_files:
    if 'csv' in num:
        if 'RH' not in num and 'LH' not in num and 'RF' not in num and 'LF' not in num and 'stats' not in num:
            all_csvs.append(num.split('/')[2].split('.')[0])

#print(all_csvs)
#And this cell does the brunt of the work.

d = {'deviation': [], 'range': [], 'rsd': [], 'amplitude': [], 'period': []}
df = pd.DataFrame(data=d)

for prefix in all_csvs:

    fparams = ff.ferret_fit_color(folder, prefix, plot=False)
    df = df.append({'deviation':fparams.ystd, 'range':fparams.yrange, 'rsd':fparams.rsd, 'amplitude':fparams.amp,
                    'pawcount':fparams.pawcount, 'pawdensity':fparams.pawdens, 'period':fparams.period,
                    'cross':fparams.cross, 'crossdensity':fparams.crossdens, 'stride':fparams.stride,
                    'stridestd':fparams.stridestd, 'run':prefix,
                    'LH range': fparams.yrangefoot['LH'], 'LH std': fparams.ystdfoot['LH'], 'LH rsd': fparams.yrsdfoot['LH'],
                    'RH range': fparams.yrangefoot['RH'], 'RH std': fparams.ystdfoot['RH'], 'LH rsd': fparams.yrsdfoot['RH'],
                    'LF range': fparams.yrangefoot['LF'], 'LF std': fparams.ystdfoot['LF'], 'LH rsd': fparams.yrsdfoot['LF'],
                    'RF range': fparams.yrangefoot['RF'], 'RF std': fparams.ystdfoot['RF'], 'LH rsd': fparams.yrsdfoot['RF']},
                    ignore_index=True)

df.to_csv('ferret_stats.csv')

stats = 'ferret_stats.csv'
aws.upload_s3(stats, '{}/{}'.format(folder, stats), bucket_name='ccurtis.data')