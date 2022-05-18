#csv test
import numpy as np
import pandas as pd 
import seaborn as sns
import matplotlib.pyplot as plt


f1 = pd.read_csv("accepted_2007_to_2018Q4.csv.gz", compression='gzip')
f2 = pd.read_csv("rejected_2007_to_2018Q4.csv.gz", compression='gzip')
acc_df = f1.sample(n = 100000)
rej_df = f2.sample(n = 100000)
print(acc_df.head().T)
acc_df[acc_df['grade'] == 'A'].describe().T
default_col = ['Does not meet the credit policy. Status:Charged Off', ' Charged Off', 'Default']
acc_df[acc_df['loan_status'].isin(default_col)].describe().T

rej_df[rej_df['grade'] == 'A'].describe().T
rej_df[rej_df['loan_status'].isin(default_col)].describe().T

nan_check1 = acc_df.isna().mean()
nan_check1 = nan_check1[nan_check1 > .9].sort_values()
nan_check1

nan_check2 = rej_df.isna().mean()
nan_check2 = nan_check2[nan_check2 > .9].sort_values()
nan_check2

acc_df = acc_df.drop(['desc','member_id'], axis=1, errors='ignore')
rej_df = acc_df.drop(['desc','member_id'], axis=1, errors='ignore')

acc_df['issue_d'] = pd.to_datetime(acc_df['issue_d'])
acc_df[['issue_d']]

acc_df['Year'] = pd.DatetimeIndex(acc_df['issue_d']).year
acc_df[['Year']]

#Do you observe different loan grade patterns in different years?
acc_df.groupby(['Year',])['funded_amnt'].agg(['mean']).plot.line()

#Do you observe different loan grade patterns for different loan purposes?
acc_df.groupby(['purpose', 'grade'])['funded_amnt'].count().nlargest(10).plot.bar()

acc_df.groupby(['Year', 'grade'])['funded_amnt'].agg(['count']).plot.line()
acc_df.groupby(['Year', 'purpose'])['purpose'].agg(['count'])
acc_df.groupby(['purpose'])['purpose'].size()

acc_df[acc_df['purpose'] == 'debt_consolidation'].groupby(['funded_amnt'])['Year'].agg(['count'])

acc_df.groupby(['fico_range_low','funded_amnt'])['funded_amnt'].agg(['count','mean','std'])
acc_df.groupby(['fico_range_high','funded_amnt'])['fico_range_high'].agg(['count','mean'])

fico_high = acc_df.groupby(['fico_range_high'])['fico_range_high'].agg(['count'])
plt.plot(fico_high)
[[fico_high]]
# NO difference ?
fico_low = acc_df.groupby(['fico_range_low'])['funded_amnt'].agg(['count'])
plt.plot(fico_low)
[[fico_low]]

#showing the mean and std of fico range between purpose of loan 
acc_df.groupby(['purpose'])['fico_range_high'].agg(['mean','std']).plot.line()
#showing the mean and std of fund amount between purpose of loan 
acc_df.groupby(['purpose'])['funded_amnt'].agg(['mean','std'])


#showing the mean and std of fico range between grade of loan 
acc_df.groupby(['grade'])['fico_range_high'].agg(['mean','std']).plot.line()
#showing the mean and std of fund amount between grade of loan ***
acc_df.groupby(['grade'])['funded_amnt'].agg(['mean','std']).plot.line()

#looking at the interest rate average by grade
acc_df.groupby(['grade'])['int_rate'].agg(['mean','std']).plot.line()