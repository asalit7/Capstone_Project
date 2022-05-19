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
#acc_df[acc_df['grade'] == 'A'].describe().T
default_col = ['Does not meet the credit policy. Status:Charged Off', ' Charged Off', 'Default']
#acc_df[acc_df['loan_status'].isin(default_col)].describe().T

#rej_df[rej_df['grade'] == 'A'].describe().T
#rej_df[rej_df['loan_status'].isin(default_col)].describe().T

nan_check1 = acc_df.isna().mean()
nan_check1 = nan_check1[nan_check1 > .9].sort_values()
nan_check1

nan_check2 = rej_df.isna().mean()
nan_check2 = nan_check2[nan_check2 > .9].sort_values()
nan_check2
#dropping na columns with all na values
acc_df = acc_df.drop(['desc','member_id'], axis=1, errors='ignore')
rej_df = acc_df.drop(['desc','member_id'], axis=1, errors='ignore')
#setting date to datetime
acc_df['issue_d'] = pd.to_datetime(acc_df['issue_d'])
acc_df[['issue_d']]
#creating a column with year
acc_df['Year'] = pd.DatetimeIndex(acc_df['issue_d']).year
acc_df[['Year']]

#Do you observe different loan grade patterns in different years?
acc_df.groupby(['Year',])['funded_amnt'].agg(['mean']).plot.line()
#amt of money for specific grades in purpose ?


#Do you observe different loan grade patterns for different loan purposes?
acc_df.groupby(['purpose', 'grade'])['funded_amnt'].count().nlargest(10).plot.bar()
#looking at grade in terms of funded amount over the years, use pivot wider
acc_df.groupby(['Year', 'grade'])['funded_amnt'].agg(['count']).plot.line()

acc_df.groupby(['Year', 'grade'])['funded_amnt'].agg(['count']).reset_index().pivot(index='Year',columns = 'grade', values = 'count').plot()
#looking at purpose count over the years 
acc_df.groupby(['Year', 'purpose'])['purpose'].agg(['count'])
acc_df.groupby(['purpose'])['purpose'].size()

acc_df[acc_df['purpose'] == 'debt_consolidation'].groupby(['funded_amnt'])['Year'].agg(['count'])

#fico range grouped low to high
acc_df.groupby(['fico_range_low','fico_range_high'])['funded_amnt'].agg(['mean','std'])
acc_df.groupby(['fico_range_high','funded_amnt'])['fico_range_high'].agg(['count','mean'])

fico_high = acc_df.groupby(['fico_range_high'])['fico_range_high'].agg(['count'])
plt.plot(fico_high)
[[fico_high]]
# NO difference ?
fico_low = acc_df.groupby(['fico_range_low'])['funded_amnt'].agg(['count'])
plt.plot(fico_low)
[[fico_low]]

#showing the mean and std of fico range between purpose of loan 
acc_df.groupby(['purpose'])['fico_range_high'].agg(['mean','std'])
#showing the mean and std of fund amount between purpose of loan 
acc_df.groupby(['purpose'])['funded_amnt'].agg(['mean','std'])


#showing the mean of fico range between grade of loan 
acc_df.groupby(['grade'])['fico_range_high'].agg(['mean']).plot.line()
#showing the mean and std of fund amount between grade of loan ***
acc_df.groupby(['grade'])['funded_amnt'].agg(['mean']).plot.line()

#looking at the interest rate average by grade
acc_df.groupby(['grade','term'])['int_rate'].agg(['mean','std'])

acc_df.groupby(['Year','grade'])['int_rate'].agg(['mean']).reset_index().pivot(index='Year',columns = 'grade', values = 'mean').plot()


#biggest difference is A grade starts lower at 36 months and G starts higher than 60 months
acc_df[acc_df['term'] == ' 60 months'].groupby(['grade'])['int_rate'].agg(['mean','std'])
acc_df[acc_df['term'] == ' 36 months'].groupby(['grade'])['int_rate'].agg(['mean','std'])
# create box plots for different grades ? 


# looking at interest rate with payment over 60 months subgrades
acc_df[acc_df['term'] == ' 60 months'].groupby(['sub_grade'])['int_rate'].agg(['mean','std'])
# looking at interest rate with payment over 36 months subgrades
#starts lower than 60 months  but lowest grade has similar prices
acc_df[acc_df['term'] == ' 36 months'].groupby(['sub_grade'])['int_rate'].agg(['mean','std'])
#mentions that misleading to compare two loans at same rate with different years 
acc_df.groupby(['Year'])['int_rate'].agg(['mean']).plot.line()


acc_df.groupby(['Year'])['dti'].agg(['mean','std'])

for col in acc_df.columns:
    print(col)


acc_df['loan_status'].unique()
#looking at loan status of funded amount within grades of defaulted vs paid off or currently paying
acc_df[acc_df['loan_status'].isin(['Does not meet the credit policy. Status:Charged Off', 'Charged Off', 'Default'])].groupby('grade')['funded_amnt'].agg(['mean']).plot.line()
acc_df[acc_df['loan_status'].isin(['Fully Paid', 'Current', 'Does not meet the credit policy. Status:Fully Paid'])].groupby('grade','loan_status')['funded_amnt'].agg(['mean']).plot.line()

acc_df[acc_df['loan_status'].isin(['Does not meet the credit policy. Status:Charged Off', 'Charged Off', 'Default'])].groupby('grade')['int_rate'].agg(['mean']).plot.line()
acc_df[acc_df['loan_status'].isin(['Fully Paid', 'Current', 'Does not meet the credit policy. Status:Fully Paid'])].groupby('grade')['int_rate'].agg(['mean']).plot.line()

acc_df[acc_df['loan_stat']]