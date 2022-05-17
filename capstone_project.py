#csv test
import numpy as np
import pandas as pd 


f1 = pd.read_csv("accepted_2007_to_2018Q4.csv.gz", compression='gzip')
f2 = pd.read_csv("rejected_2007_to_2018Q4.csv.gz", compression='gzip')
acc_df = f1.sample(n = 100000)
rej_df = f2.sample(n = 100000)
print(acc_df.head().T)
acc_df[acc_df['grade'] == 'A'].describe().T
default_col = ['Does not meet the credit policy. Status:Charged Off', ' Charged Off', 'Default']
acc_df[acc_df['loan_status'].isin(default_col)].describe().T