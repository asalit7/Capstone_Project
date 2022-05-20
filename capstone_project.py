#csv test
import numpy as np
import pandas as pd 
import seaborn as sns
import matplotlib.pyplot as plt


f1 = pd.read_csv("accepted_2007_to_2018Q4.csv.gz", compression='gzip')
f2 = pd.read_csv("rejected_2007_to_2018Q4.csv.gz", compression='gzip')
f1.shape
acc_df = f1.sample(n = 100000)
rej_df = f2.sample(n = 100000)
print(acc_df.head().T)
#acc_df[acc_df['grade'] == 'A'].describe().T
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
#creating a column with year
acc_df['Year'] = pd.DatetimeIndex(acc_df['issue_d']).year

#cleaning up loan status 
acc_df.loc[acc_df['loan_status'] == ('Does not meet the credit policy. Status:Charged Off'), 'loan_status'] = 'Charged Off'
acc_df.loc[acc_df['loan_status'] == ('Does not meet the credit policy. Status:Fully Paid'),'loan_status'] = 'Fully Paid'

#graphing funded amount over the years
acc_df.groupby(['Year'])['funded_amnt'].agg(['mean']).plot.line()
avg_fnd = sns.lineplot(data = acc_df, x='Year', y='funded_amnt', estimator=np.mean).set(xlabel ="Year", ylabel ="Funded Amount", title="Loan Amount Over the Years")
#amt of money for specific grades in purpose ? ------------------------------------------------------
#looking purpose counts over the years
acc_df.groupby(['Year','purpose'])['funded_amnt'].agg(['count']).reset_index().pivot(index='Year',columns = 'purpose', values = 'count').plot()

sns.lineplot(data=acc_df,x='funded_amnt',y='purpose')
acc_df.groupby(['purpose'])['funded_amnt'].agg(['count'])


#looking at the most used reason for loans over the years
acc_df.groupby(['Year','purpose'])['funded_amnt'].count().nlargest(15).plot.barh()
#Do you observe different loan grade patterns for different loan purposes?
acc_df.groupby(['purpose', 'grade'])['funded_amnt'].count().nlargest(10).plot.bar()
acc_df.groupby(['purpose'])['funded_amnt'].count().nlargest(10).plot.bar()
#looking purpose counts over the years
acc_df.groupby(['Year','purpose'])['funded_amnt'].agg(['count']).reset_index().pivot(index='Year',columns = 'purpose', values = 'count').plot()

#looking at grade in terms of funded amount over the years, use pivot wider
acc_df.groupby(['Year', 'grade'])['funded_amnt'].agg(['count']).reset_index().pivot(index='Year',columns = 'grade', values = 'count').plot()
#looking at purpose count size
acc_df.groupby(['purpose'])['purpose'].size()

#fico range grouped low to high
acc_df.groupby(['Year','fico_range_low','fico_range_high'])['funded_amnt'].agg(['mean']).reset_index()
acc_df.groupby(['fico_range_high','funded_amnt'])['fico_range_high'].agg(['count','mean'])

fico_high = acc_df.groupby(['fico_range_high'])['fico_range_high'].agg(['count'])
plt.plot(fico_high)

fico_low = acc_df.groupby(['fico_range_low'])['funded_amnt'].agg(['count'])
grade_check = acc_df.groupby(['Year','grade'])['t_amnt']
plt.plot(fico_low)

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


#biggest difference is A grade starts lower at 36 months and G starts higher than 60 months
acc_df[acc_df['term'] == ' 60 months'].groupby(['grade'])['int_rate'].agg(['mean','std'])
acc_df[acc_df['term'] == ' 36 months'].groupby(['grade'])['int_rate'].agg(['mean','std'])
# create box plots for different grades ? 

#pivoting a table looking at how int rate has changed over the years with grade
acc_df.groupby(['Year','grade'])['int_rate'].agg(['mean']).reset_index().pivot(index='Year',columns = 'grade', values = 'mean').plot()

# looking at interest rate with payment over 60 months subgrades
acc_df[acc_df['term'] == ' 60 months'].groupby(['Year','grade'])['int_rate'].agg(['mean']).reset_index().pivot(index='Year',columns = 'grade', values = 'mean').plot()
# looking at interest rate with payment over 36 months subgrades
#starts lower than 60 months  but lowest grade has similar prices
acc_df[acc_df['term'] == ' 36 months'].groupby(['Year','grade'])['int_rate'].agg(['mean']).reset_index().pivot(index='Year',columns = 'grade', values = 'mean').plot()

#term doesnt affect the int rate average over the years too much

#mentions that misleading to compare two loans at same rate with different years 
acc_df.groupby(['Year'])['int_rate'].agg(['mean']).plot.line()


acc_df.groupby(['Year'])['dti'].agg(['mean','std'])

for col in acc_df.columns:
    print(col)

acc_df['loan_status'].unique()
#looking at loan status of funded amount within grades of defaulted vs paid off or currently paying
acc_df[acc_df['loan_status'].isin(['Does not meet the credit policy. Status:Charged Off', 'Charged Off', 'Default'])].groupby('grade')['int_rate'].agg(['count'])
acc_df[acc_df['loan_status'].isin(['Fully Paid', 'Current', 'Does not meet the credit policy. Status:Fully Paid'])].groupby('grade')['int_rate'].agg(['count'])
acc_df.groupby('grade')['int_rate'].agg(['count'])
acc_df[acc_df['loan_status'].isin(['Does not meet the credit policy. Status:Charged Off', 'Charged Off', 'Default'])].groupby('grade')['int_rate'].agg(['mean']).plot.line()
acc_df[acc_df['loan_status'].isin(['Fully Paid', 'Current', 'Does not meet the credit policy. Status:Fully Paid'])].groupby('grade')['int_rate'].agg(['mean']).plot.line()

#creating variables of defaulted loans
status_sum = acc_df.groupby('grade')['loan_status'].count().sum()
def_grade_stat = acc_df[acc_df['loan_status'].isin(['Does not meet the credit policy. Status:Charged Off', 'Charged Off', 'Default'])].groupby('grade').size()
#calculating percentages and graphing defaulted loans
def_grade_perc = (def_grade_stat/status_sum)*100
def_grade_perc.plot.line()

#creating variables of paid loans
paid_grade_stat = acc_df[acc_df['loan_status'].isin(['Fully Paid', 'Current', 'Does not meet the credit policy. Status:Fully Paid'])].groupby('grade').size()
#calculated percentage of paid loans and graphed it
paid_grade_perc = (paid_grade_stat/status_sum)*100
paid_grade_perc.plot.line()
#looking at total loan status percentages by grade
total_stat = (acc_df.groupby('grade')['loan_status'].count()/status_sum)*100
total_stat.plot.line()

acc_df[['total_pymnt']]
acc_df[['funded_amnt']]
#creating a variable for profit of each row 
acc_df.groupby('grade')['total_pymnt'].agg(['mean'])
acc_df['profit'] = acc_df['total_pymnt'] - acc_df['funded_amnt']
acc_df['profit_perc'] = (acc_df['total_pymnt'] - acc_df['funded_amnt'])/acc_df['funded_amnt']
#graphing the profit by Year and grade
acc_df.groupby(['Year','grade'])['profit'].agg(['mean']).reset_index().pivot(index='Year',columns = 'grade', values = 'mean').plot()

#graphing by profit for each year and grade
acc_df.groupby(['Year','grade'])['profit'].agg(['mean']).reset_index().pivot(index='Year',columns = 'grade', values = 'mean').plot()
acc_df.groupby(['Year','grade'])['profit_perc'].agg(['count']).reset_index().pivot(index='Year',columns = 'grade', values = 'count').plot()
acc_df.groupby(['Year','grade'])['profit_perc'].agg(['mean'])
acc_df.groupby(['Year','grade'])['profit_perc'].agg(['count'])

#creating variable for the average interest rate by grade
grade_rate = acc_df.groupby(['grade'])['int_rate'].agg(['mean']).reset_index()

#merging grade_rate to original dataset to have a mean for int_rates
acc_df = pd.merge(acc_df,grade_rate,how='left',on=['grade'])
acc_df = acc_df.rename(columns={'mean':'grade_rate_mean'})

#graphing interest grade means with profit over the years
acc_df[acc_df['loan_status'] != 'Current'].groupby(['Year','grade_rate_mean'])['profit'].agg(['mean']).reset_index().pivot(index='Year',columns = 'grade_rate_mean', values = 'mean').plot()
acc_df[acc_df['loan_status'] != 'Current'].groupby(['Year','grade_rate_mean'])['profit'].agg(['count']).reset_index().pivot(index='Year',columns = 'grade_rate_mean', values = 'count').plot()
acc_df.groupby(['Year','grade_rate_mean'])['profit'].agg(['count']).reset_index().pivot(index='Year',columns = 'grade_rate_mean', values = 'count').plot()

#looking at initial graph of purpose with profit mean
acc_df.groupby(['Year', 'purpose'])['profit'].agg(['mean']).reset_index().pivot(index='Year',columns = 'purpose', values = 'mean').plot()

acc_df[acc_df['Year'].isin([2016.0,2017.0,2018.0])].groupby(['grade'])['profit'].agg(['mean'])
#looking at values in different loan status's
acc_df[acc_df['loan_status'].isin(['Late (31-120 days)','Late (16-30 days)'])].groupby(['Year'])['profit'].agg(['mean']).plot.line()
acc_df[acc_df['loan_status'].isin(['Fully Paid', 'Does not meet the credit policy. Status:Fully Paid'])].groupby(['Year'])['profit'].agg(['mean']).plot.line()

#looking at profit in terms of different terms
acc_df[acc_df['loan_status'] != 'Current'].groupby(['Year','term'])['profit'].agg(['mean']).reset_index().pivot(index='Year',columns = 'term', values = 'mean').plot()


#looking into profit by grade over the years of loans that are not currently paying back
acc_df[acc_df['loan_status'].isin(['Default','Charged Off','Fully Paid'])].groupby(['Year','grade'])['profit'].agg(['mean']).reset_index().pivot(index='Year',columns = 'grade', values = 'mean').plot()

#creating a mask of the dataframe without the current loans
loan_mask = acc_df[acc_df['loan_status'] != 'Current']
#count plot of the count of purpose loans
sns.countplot(data=acc_df, y='purpose').set(xlabel ="Count", ylabel = "Purpose of Loan",title='Purpose of Loan Count')
#bar plot of average funds 
sns.lineplot(data=loan_mask,x='grade',y='funded_amnt')
sns.barplot(data=loan_mask,x='grade',y='interest_rates')

acc_df[acc_df['loan_status'] != 'Current'].groupby(['Year','grade'])['profit'].agg(['mean']).reset_index().pivot(index='Year',columns = 'grade', values = 'mean').plot()


loan_mask = acc_df[acc_df['loan_status'] != 'Current']
loan_mask

acc_df[acc_df['loan_status'] != 'Current'].groupby(['Year','term'])['profit'].agg(['mean']).reset_index().pivot(index='Year',columns = 'grade_rate_mean', values = 'mean').plot()
