#csv test
import numpy as np
import pandas as pd 
import seaborn as sns
import matplotlib.pyplot as plt


acc_df = pd.read_csv("accepted_2007_to_2018Q4.csv.gz", compression='gzip')

acc_df.shape

print(acc_df.head().T)

#checking the amount of na values present in columns and sorting it from desc
nan_check1 = acc_df.isna().mean()
nan_check1 = nan_check1[nan_check1 > .9].sort_values()
nan_check1

#dropping na columns with all na values
acc_df = acc_df.drop(['desc','member_id'], axis=1, errors='ignore')
#setting date to datetime
acc_df['issue_d'] = pd.to_datetime(acc_df['issue_d'])
#creating a column with year
acc_df['Year'] = pd.DatetimeIndex(acc_df['issue_d']).year

#cleaning up loan status 
acc_df.loc[acc_df['loan_status'] == ('Does not meet the credit policy. Status:Charged Off'), 'loan_status'] = 'Charged Off'
acc_df.loc[acc_df['loan_status'] == ('Does not meet the credit policy. Status:Fully Paid'),'loan_status'] = 'Fully Paid'

#creating a column for profits of rows
acc_df['profit'] = acc_df['total_pymnt'] - acc_df['funded_amnt']

#creating fico range into one variable
acc_df['fico_range'] = (acc_df['fico_range_high'] + acc_df['fico_range_low'])/2

#creating a mask of the dataframe without the current loans
loan_mask = acc_df[acc_df['loan_status'] != 'Current']
#creating a mask of data frame with only current loans
curr_loan_mask = acc_df[acc_df['loan_status'] == 'Current']

#graphing funded amount over the years
acc_df.groupby(['Year'])['funded_amnt'].agg(['mean']).plot.line()
avg_fnd = sns.lineplot(data = acc_df, x='Year', y='funded_amnt', estimator=np.mean).set(xlabel ="Year", ylabel ="Funded Amount", title="Loan Funded Amount Over the Years")

#-------------- Grade graphs and information

#create interest rates with grade box plots
non_current_mask = acc_df[acc_df['loan_status'] != 'Current']
sns.boxplot(data=non_current_mask,x='grade', y='int_rate', order=['A','B','C','D','E','F','G']).set(xlabel = 'Grade', ylabel='Interest Rate', title='Interest Rates by Grade')

yr_gr_funds = acc_df.loc[acc_df['loan_status'] != 'Current'].groupby(['Year','grade'])['funded_amnt'].agg(['mean']).reset_index().pivot(index='Year',columns = 'grade', values = 'mean')
yr_gr_funds.plot.line(xlabel = 'Year',ylabel='Funded Amount', title = 'Grade Average Funded Amounts Change Over Years')

#looking at grade in terms of funded amount over the years, use pivot wider
acc_df.groupby(['Year', 'grade'])['funded_amnt'].agg(['count']).reset_index().pivot(index='Year',columns = 'grade', values = 'count').plot()

#creating a count plot for the amount of grades 
sns.countplot(data = acc_df, x='grade',order=['A','B','C','D','E','F','G']).set(xlabel ="Grade", ylabel = "Count",title='Count of Each Grade')

#creating grade box plots by fico score
sns.boxplot(data=acc_df,x='grade', y='fico_range', order=['A','B','C','D','E','F','G']).set(xlabel = 'Grade', ylabel='Fico Score', title='Fico Score by Grade')

#looking at the interest rate average by grade
acc_df.groupby(['grade','term'])['int_rate'].agg(['mean'])

#biggest difference is A grade starts lower at 36 months and G starts higher than 60 months
acc_df[acc_df['term'] == ' 60 months'].groupby(['grade'])['int_rate'].agg(['mean','std'])
acc_df[acc_df['term'] == ' 36 months'].groupby(['grade'])['int_rate'].agg(['mean','std'])

#showing grade by interest rate
acc_df.groupby('grade')['fico_range'].agg(['mean'])
#bar plot of average funds 
sns.lineplot(data=loan_mask,x='grade',y='funded_amnt')
#looking at loan status of funded amount within grades of defaulted vs paid off or currently paying
defaulted_bar = acc_df[acc_df['loan_status'].isin(['Does not meet the credit policy. Status:Charged Off', 'Charged Off', 'Default'])].groupby('grade')['int_rate'].agg(['count'])
defaulted_bar.plot.bar(xlabel="Grade",ylabel="Count",title='Count of Defaulted Borrowers')

#showing a bar graph of non_defaulted borrowers count
non_defaulted_bar = acc_df[acc_df['loan_status'].isin(['Fully Paid', 'Does not meet the credit policy. Status:Fully Paid'])].groupby('grade')['int_rate'].agg(['count'])
non_defaulted_bar.plot.bar(xlabel="Grade",ylabel="Count",title='Count of Non-Defaulted Borrowers')

# Purpose -----------
#count plot of the count of purpose loans
sns.countplot(data=acc_df, y='purpose').set(xlabel ="Count", ylabel = "Purpose of Loan",title='Purpose of Loan Count')
#creating a separate dataframing looking at the highest count purpose rows
purpose_mask = acc_df.loc[acc_df['purpose'].isin(['credit_card','debt_consolidation','home_improvement','other'])]
# looking into profit over the years
purpose_profit = purpose_mask[purpose_mask['loan_status'] != 'Current'].groupby(['Year', 'purpose'])['profit'].agg(['mean']).reset_index().pivot(index='Year',columns = 'purpose', values = 'mean')
purpose_profit.plot.line(xlabel='Year', ylabel= 'Average Profit', title = 'Average Profit Amount by Purpose')
# looking into purpose of funding amnts of the years
purpose_fund = purpose_mask[purpose_mask['loan_status'] != 'Current'].groupby(['Year', 'purpose'])['funded_amnt'].agg(['mean']).reset_index().pivot(index='Year',columns = 'purpose', values = 'mean')
purpose_fund.plot.line(xlabel='Year', ylabel= 'Average Loan Fund Amount', title = 'Average Loan Fund Amount by Purpose')
# looking into the grades in respect to purpose of profits
purpose_mask[purpose_mask['loan_status'] != 'Current'].groupby(['grade', 'purpose'])['profit'].agg(['mean']).reset_index().pivot(index='grade',columns = 'purpose', values = 'mean').plot()

# Years----------------

#pivoting a table looking at how int rate has changed over the years with grade
acc_df.groupby(['Year','grade'])['int_rate'].agg(['mean']).reset_index().pivot(index='Year',columns = 'grade', values = 'mean').plot()

#mentions that misleading to compare two loans at same rate with different years 
acc_df.groupby(['Year'])['int_rate'].agg(['mean']).plot.line()

#graphing by profit for each year and grade
yr_gr_profit = acc_df.loc[acc_df['loan_status'] != 'Current'].groupby(['Year','grade'])['profit'].agg(['mean']).reset_index().pivot(index='Year',columns = 'grade', values = 'mean')
yr_gr_profit.plot.line(xlabel = 'Year',ylabel='Profit', title = 'Grade Profits Change Over Years')

#looking at profit in terms of different terms
term_profit = acc_df[acc_df['loan_status'] != 'Current'].groupby(['Year','term'])['profit'].agg(['mean']).reset_index().pivot(index='Year',columns = 'term', values = 'mean')
term_profit.plot.line(xlabel = 'Year',ylabel='Profit', title = 'Term Profits Change Over Years')

#looking at loan status of funded amount within grades of defaulted vs paid off or currently paying
defaulted_bar = acc_df[acc_df['loan_status'].isin(['Does not meet the credit policy. Status:Charged Off', 'Charged Off', 'Default'])].groupby('grade')['int_rate'].agg(['count'])
defaulted_bar.plot.bar(xlabel="Grade",ylabel="Count",title='Count of Defaulted Borrowers')

#showing a bar graph of non_defaulted borrowers count
non_defaulted_bar = acc_df[acc_df['loan_status'].isin(['Fully Paid', 'Does not meet the credit policy. Status:Fully Paid'])].groupby('grade')['int_rate'].agg(['count'])
non_defaulted_bar.plot.bar(xlabel="Grade",ylabel="Count",title='Count of Fully Paid Borrowers')

#creating a variable for profit of each row 
acc_df.groupby('grade')['total_pymnt'].agg(['mean'])
#graphing the profit by Year and grade
acc_df.groupby(['Year','grade'])['profit'].agg(['mean']).reset_index().pivot(index='Year',columns = 'grade', values = 'mean').plot()

#looking at values in different loan status's
acc_df[acc_df['loan_status'].isin(['Late (31-120 days)','Late (16-30 days)'])].groupby(['Year'])['profit'].agg(['mean']).plot.line()
acc_df[acc_df['loan_status'].isin(['Fully Paid', 'Does not meet the credit policy. Status:Fully Paid'])].groupby(['Year'])['profit'].agg(['mean']).plot.line()


