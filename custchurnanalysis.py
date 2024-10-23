
import streamlit as st 
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt 
import seaborn as sns 
import plotly.express as px


st.set_page_config(
    page_title="Data Dashboard",
    page_icon="🦄",
    # layout="wide",
    # initial_sidebar_state="expanded",
    # menu_items={
    #     'Get Help': 'https://www.extremelycoolapp.com/help',
    #     'Report a bug': "https://www.extremelycoolapp.com/bug",
    #     'About': "# This is a header. This is an *extremely* cool app!"
    # }
)
st.title("Customer Churn Data Dashboard ")




# file upload
# uploaded_file = st.file_uploader("Choose a CSV file", type="csv")


# create a dataframe for the uploaded CSV
df = pd.read_csv("TelecomChurn.csv")
  
st.subheader("Data Preview")
st.write( df.head() )
  
# -------- churn counts
st.divider()
churn_count = df['churn'].value_counts()
# normalized
churn_counts_normal = df['churn'].value_counts(normalize=True)
# st.table( churn_counts_normal )
  
col1, col2, col3, col4  = st.columns(4)
col1.metric("Customer Churn", churn_count.iloc[1], 3)
k = f"{churn_counts_normal.iloc[1] *100:.1f}%"
col2.metric("Customer Churn %", k , 3)
m = f"{churn_counts_normal.iloc[0] *100:.1f}%"
col3.metric("Customer Non-Churn", churn_count.iloc[0] , -3)
col4.metric("Customer Non-Churn %", m, -3)

st.divider()
  
  
  
  
  
# ------ df.describe() = summary
st.subheader("Data Summary Statistics")
st.write( df.describe() )

st.divider()
  
# df['churn'] = df['churn'].astype('int64')
# st.table(df.loc[0:5,"total intl calls":"churn"] )
# desc = df.describe()
# st.table( desc )
# desc2= df.describe(include=['object','bool'])
# st.table( desc2 )
  
# Group data by 'churn' and compute the mean 
group_churn_mean = df.groupby('churn')['customer service calls'].mean()

# st.subheader("Average Customer Service Calls")


st.markdown("## Average :orange[Churn] Customers")
col1, col2 = st.columns(2)
churn_totaldaymin = df[df['churn']==1] ['total day minutes'].mean()
non_churn_totaldaymin = df[df['churn']==0] ['total day minutes'].mean()
delta_totaldaymin = f"{churn_totaldaymin - non_churn_totaldaymin :.2f}"
totaldaymin = f"{churn_totaldaymin:.2f}"
col1.metric("Total Day Call Minutes", totaldaymin, delta_totaldaymin )
col2.write("The average difference between churn customers and ongoing customers is 31.74 minutes. Churn customers make more total day calls. ")
  

churn_totalnightmin = df[df['churn']==1] ['total night minutes'].mean()
non_churn_totalnightmin = df[df['churn']==0] ['total night minutes'].mean()
delta_totalnightmin = f"{churn_totalnightmin - non_churn_totalnightmin :.2f}"
totalnightmin = f"{churn_totalnightmin:.2f}"
col3, col4 = st.columns(2)
col3.metric("Total Night Call Minutes",totalnightmin , delta_totalnightmin )
col4.write("The total average difference for night call minutes between churn customer and ongoing customers is 5 minutes.")


churn_total_intlcalls = df[df['churn']==1] ['total intl calls'].mean()
non_churn_total_intlcalls = df[df['churn']==0] ['total intl calls'].mean()
delta_total_intlcalls = f"{churn_total_intlcalls - non_churn_total_intlcalls :.2f}"
totalintlmin = f"{churn_total_intlcalls:.2f}"
col5,col6 = st.columns(2)
col5.metric("Total International Calls", totalintlmin , delta_total_intlcalls)
col6.write("The total average difference for international calls between churn customers and ongoing customers is 0.37 more calls compared to non-churn customers.")


churn_total_custcalls = df[df['churn']==1] ['customer service calls'].mean()
non_churn_total_custcalls = df[df['churn']==0] ['customer service calls'].mean()
delta_total_custcalls = f"{churn_total_custcalls - non_churn_total_custcalls :.2f}"
total_custcalls = f"{churn_total_custcalls:.2f}"
col7,col8 = st.columns(2)
col7.metric("Total Customer Service calls", total_custcalls , delta_total_custcalls)
col8.write("The total average difference for customer service calls is 0.78 more than ongoing customers.")

st.divider()







# col3, col4 = st.columns(2)
  
# col3.metric("by Customers", round( group_churn_mean.iloc[0],2)  )
# col4.metric("by Churn Customers", round( group_churn_mean.iloc[1],2)  )
  
  
# Count the number of churners and non-churners by State 
group_churn_by_state_count = df.groupby('state')['churn'].value_counts().sort_values(ascending=False)
# st.table(group_churn_by_state_count.head() )
  

  

total_day_charge = df.sort_values(by='total day charge', ascending=False)
# st.table( total_day_charge.head() )

churn_total_day_charge = df.sort_values(by=['churn','total day charge'],ascending=False).head()
# st.subheader("Churn by total day charges")
# st.table( churn_total_day_charge.head() )



columns_to_show = ['total day minutes', 'total eve minutes', 
                   'total night minutes']
groupby_agg_columns = df.groupby(['churn'])[columns_to_show].mean()
st.markdown("### :orange[Churn Customers] grouped by _average_ for total minutes for parts of the day")
st.table( groupby_agg_columns )
st.write("Churn customers take up more total daytime and evening minutes than regular customers.")

groupby_agg_columns = df.groupby(['churn'])[columns_to_show].max()
st.markdown("### :orange[Churn Customers] grouped by _maximum_ values for total minutes")
st.table(groupby_agg_columns)
st.write("Churn customers have the highest values for maximum day and evening calls compared to regular customers.")


st.markdown("### Customer :orange[Churn] call minutes by percentiles")
groupby_churners_percentiles = df.groupby(['churn'])[columns_to_show].describe(percentiles=[])
st.table( groupby_churners_percentiles )
st.write("Churn customers summary statistics on times of day minutes for calls.")




st.markdown("### Customers by Account length by Custer Service calls")
fig = px.scatter(data_frame= df, x= "account length", y= "customer service calls", color="churn")
st.plotly_chart(fig, key="churn", on_select="rerun")

st.subheader("Boxplot of Customer Service calls")
fig = px.box(data_frame=df, y="customer service calls", color="churn")
st.plotly_chart(fig, key="churn2")
st.write("The boxplot above shows that Churn customers do make the largest volume of customer service calls.")




st.divider()





st.subheader("International Plans")

churner_intl_plan = pd.crosstab(df['churn'], df['international plan'], margins=True)
st.write("Customers vs Churn by International Plans with column totals")
st.table( churner_intl_plan )

col9,col10,col11,col12 = st.columns(4)
crosstab = pd.crosstab(df['churn'], df['international plan'], normalize= True)

cust_no_intl_plan = f"{crosstab.iloc[0][0]*100:.1f}%"
cust_intl_plan = f"{crosstab.iloc[0][1]*100:.1f}%"

col9.metric("Customers without ",cust_no_intl_plan)
col10.metric("Customers with ", cust_intl_plan)
churn_intl_plan = f"{crosstab.iloc[1][1]*100:.1f}%"
churn_no_intl_plan = f"{crosstab.iloc[1][0]*100:.1f}%"
col11.metric("Churn with Int'l Plans", churn_intl_plan )

col12.metric("Churn without Int'l Plans",churn_no_intl_plan)

st.divider()








st.markdown("## :orange[Churn] Customers by Voice Mail Plan")
churn_voicemailplan = df[df['churn'] ==1]['voice mail plan'].value_counts()




# st.write( churn_voicemailplan)
# st.table( just_churn.head() )

# st.bar_chart(churn_voicemailplan, x= x, y= y )
st.bar_chart(churn_voicemailplan, horizontal=False, width=5, color=['#ff1a8c'])
st.write("Out of the Churn customers, the majority did not have voice mail plans.")

# st.divider()








# accts = df[ df['churn']==1 ][['account length','state']].sort_values( ascending= False, by='account length')#.value_counts()
# st.dataframe( accts.head() )


# accts_crosstab = pd.crosstab(df['churn'], ['account length'])
# st.table( accts_crosstab )

# st.subheader("Account length")
# col13,col14 = st.columns(2)
# col13.metric("Average Customer Account Length", accts_crosstab.iloc[0] )
# col14.metric("Average Churn Account Length", accts_crosstab.iloc[1] )







# churner_serv_calls = pd.crosstab(df['churn'], df['customer service calls'], margins=True)
# st.table( churner_serv_calls )
# print(churner_serv_calls)






st.divider()

st.subheader("Customer account length")
fig = px.violin(data_frame=df, x="account length" , color="churn", points="all")
st.plotly_chart(fig, key="account length")
# max_vals = df.apply(np.max)
# st.table( max_vals)


# st.subheader("Top 10 Customer Churn by State ")
# grouped_state_churners = df.groupby('state')['churn'].value_counts().iloc[1::2].sort_values(ascending=False) 
# st.table( grouped_state_churners )

# c= st.container()
# c.metric("## NJ and Texas", grouped_state_churners.loc['NJ'] )

# df[df[df['churn'] ==1]]


st.divider()

st.subheader("Customers by State and account length")
state_areacodes = df.loc[0:10, 'state': 'churn'].sort_values(ascending=False,by='account length')
# st.data_editor( state_areacodes )

fig = px.bar(data_frame=df, x= "state", y= "account length", color="churn")
st.plotly_chart(fig, key="state")
st.write("The state of West Virginia has a high count of churners, along with the states of New York and Minnesota. ")



state_churn = df.groupby(['state'])['churn'].agg([np.mean]).sort_values(by='mean', ascending=False).T
st.subheader("The average Churn by State")
st.table( state_churn )


  
# - filter dataframe columns
# st.subheader("Filter Data")
# columns = df.columns.tolist()
# selected_column = st.selectbox("Select column to filter by", columns)
# unique_values = df[selected_column].unique()
# selected_value = st.selectbox("Select value", unique_values)
  
# # display filtered data 
# filtered_df = df[df[selected_column] == selected_value]
# st.write( filtered_df )
  

# features = ['total day minutes', 'total intl calls']
# st.bar_chart( df[features].iloc[0] )


st.divider()

# fig = px.bar(data_frame= df, x= "account length", y= "customer service calls", color="churn")




st.subheader("Customers Type by Customer Service Calls")
columns_to_show2 = ['customer service calls']
cust_calls = df.groupby(['churn'])[columns_to_show2].describe(percentiles=[])
st.table( cust_calls)



st.markdown(""" 
            ## Summary 
It looks like :orange[customers who do churn] end up leaving more customer service calls 
compared to customers. Both customers and churners mostly have around the same duration 
for length of accounts. Churners do not largely have voice mail plans nor international 
call plans.

This type of information is really useful in better 
understanding the drivers of churn. 

""")





















# ------------------------------------------------------------------------------
# credit to https://github.com/harshbg/Telecom-Churn-Data-Analysis for dataset

# df = pd.read_csv("TelecomChurn.csv")

# head = df.head()
# print( head )

# columns = df.columns.tolist()
# print("\n", columns )

# info = df.info()
# print( info )


# print(f"\nchurn False: {churn_count.iloc[0] }  churn True: {churn_count.iloc[1]}"  )

# To group data by Churn and compute the mean to find out 
# if churners make more customer service calls than non-churners:

# Group data by 'churn' and compute the mean 
# group_churn_mean = df.groupby('churn')['customer service calls'].mean()
# print(f"\nChurn by Customer Service call mean:\n  False  {group_churn_mean.iloc[0]}  True  {group_churn_mean.iloc[1]}") 
# churners seem to make more customer service calls than non-churners.


# To find out if one State has more churners compared to another.
# Count the number of churners and non-churners by State 
# group_churn_by_state_count = df.groupby('state')['churn'].value_counts()#.sort_values(ascending=False)
# print(f"\n Churn by State count: \n {group_churn_by_state_count}") 

# x = group_churn_by_state_count.sort_values().loc[['CA','AZ','NY','WV']].iloc[0::2]
# grouped_state_churners = df.groupby('state')['churn'].value_counts().iloc[1::2].sort_values(ascending=False) 
# print(f"\nTop 10 State Churners: \n{grouped_state_churners[0:10]}" )


# Exploring Data Visualizations : To understand how variables are distributed.
# Import matplotlib and seaborn 
# import matplotlib.pyplot as plt 
# import seaborn as sns 
  
# Visualize the distribution of 'Total day minutes' 
# plt.hist(df['total day minutes'], bins = 100) 
  
# Display the plot 
# plt.show() 
# visualize the difference in Customer service calls between churners and non-churners
# Create the box plot 
# sns.boxplot(x = 'churn', 
#             y = 'customer service calls', 
#             data = df, 
#             # sym = "",                   
#             hue = "international plan")  
# Display the plot 
# plt.show() 




# data preprocessing

# Data Preprocessing for Telco Churn Dataset

# Many Machine Learning models make certain assumptions about how the data is distributed. 
# Some of the assumptions are as follows:

# The features are normally distributed
# The features are on the same scale
# The datatypes of features are numeric

# In telco churn data, Churn, Voice mail plan, and, International plan, 
# in particular, are binary features that can easily be converted into 0’s and 1’s.



# print("--"*35,'\n')

# print( df.info() )
# Changing the column type
# df['churn'] = df['churn'].astype('int64')


# desc = df.describe()
# print( desc )


# include parameter for statistics on non-numerical features
# desc2= df.describe(include=['object','bool'])
# print(f"describe \n{desc2}\n" )


# value_counts method to have a look at the distribution and proportion
# churn_counts = df['churn'].value_counts()
# print(f"churn counts \n{churn_counts} \n"  )


# normalize the counts
# churn_counts_normal = df['churn'].value_counts(normalize=True)
# print(f"normalized churn counts \n{churn_counts_normal} \n")



# sort
# total_day_charge = df.sort_values(by='total day charge', ascending=False).head()
# print(f"Total day charge: \n{total_day_charge} \n")


# churn_total_day_charge = df.sort_values(by=['churn','total day charge'],ascending=False).head()
# print(f"Churn by total day charge\n{churn_total_day_charge} \n")


# indexing
# df_mean = df['churn'].mean()
# print(f"the churn average: {df_mean} \n")


# churn_avg = df[df['churn'] ==1].mean()
# churn_avg = df[df['churn'] ==1]
# print(f"churn avg: \n{churn_avg} \n")


# churn_totaldaymin = df[df['churn']==1] ['total day minutes'].mean()
# print(f"churn total day minutes: \n {churn_totaldaymin} \n")

# cond1= (df['churn']==0)
# cond2 = (df['international plan'] =='no')
# nonchurn_no_intlplan = df[ cond1 & cond2]['total intl minutes'].max()
# print(f"non-churn w/o intl plan max intl min: \n{nonchurn_no_intlplan} \n")

# state_areacodes = df.loc[0:10, 'state': 'area code'].sort_values(ascending=False,by='account length')
# print(f"top 10 account length by state:\n{state_areacodes}\n")

# state_totalcalls = df.loc[0:10, 'state':'total day calls'][::2].sort_values(ascending=False,by='account length')
# print(f"top 10 total calls by state:\n{state_totalcalls}\n")


# row1 = df[:1]
# print(f"row 1 \n{row1} \n" )
# last_row = df[-1:]
# print(f"last row \n{last_row} \n" )



# get the max of all rows, columns using appy()
# max_vals = df.apply(np.max)
# print(f"max values: \n{max_vals} \n ")

# select cells with specific conditions
# state_h = df[df['state'].apply(lambda state: state[0] == 'H')].head()
# print( state_h)


#  replace values 
d = {'no' : False, 'yes' : True}
df['international plan'] = df['international plan'].map(d)
df['voice mail plan'] = df['voice mail plan'].map(d)
# print(  df.iloc[0:5,3:8] )
# print( df.iloc[0:5,3:8] )


# grouping 
columns_to_show = ['total day minutes', 'total eve minutes', 
                   'total night minutes']

# groupby_churners_percentiles = df.groupby(['churn'])[columns_to_show].describe(percentiles=[])
# print(f"\ngroup by churners percentiles: \n{groupby_churners_percentiles} \n")

# groupby_agg_columns = df.groupby(['churn'])[columns_to_show].max([np.mean, np.std, np.min, np.max])
# groupby_agg_columns = df.groupby(['churn'])[columns_to_show].mean()
# print(f"\nchurn group by avg for columns: \n {groupby_agg_columns} \n")


# groupby_agg_columns = df.groupby(['churn'])[columns_to_show].std()
# print(f"\nchurn group by std for columns: \n {groupby_agg_columns} \n")

# groupby_agg_columns = df.groupby(['churn'])[columns_to_show].min()
# print(f"\nchurn group by min for columns: \n {groupby_agg_columns} \n")

# groupby_agg_columns = df.groupby(['churn'])[columns_to_show].max()
# print(f"\nchurn group by max for columns: \n {groupby_agg_columns} \n")



# crosstab
# crosstab = pd.crosstab(df['churn'], df['international plan'])
# print(f"crosstab method for intl plan: \n {crosstab} \n")

# crosstab = pd.crosstab(df['churn'], df['voice mail plan'])
# print(f"crosstab method for voice mail plan: \n {crosstab} \n")

# crosstab = pd.crosstab(df['churn'], df['international plan'], normalize= True)
# print(f"Normalized crosstab method for intl plan: \n {crosstab} \n")


#------------- pivot table
# values - a list of variables to calculate statistics for,
# index – a list of variables to group data by,
# aggfunc — what statistics we need to calculate for groups - e.g sum, mean, maximum, minimum or something else.

# piv_table = df.pivot_table(['total day calls', 'total eve calls', 'total night calls'], ['area code'], aggfunc='mean')
# print(f"pivot table for columns: \n{piv_table} \n")



# add total_calls column to dataframe
# df['total_charge'] = df['total day charge'] + df['total eve charge'] + \
#     df['total night charge'] + df['total intl charge']

# print( df.loc[0:5,'churn':'total_charge'] )


#  drop columns  axis=1  axis=0 rows    inplace=True  for changes to stick
# del_cols = df.drop( 'total_charge', axis=1, inplace=True)
# print(del_cols,'\n', df.head() )

# crosstab for intl plan
# churner_intl_plan = pd.crosstab(df['churn'], df['international plan'], margins=True)
# print(f"\ncrosstab churners intl plan margins: \n {churner_intl_plan} \n")


# print( 2664/2850)



# churner_serv_calls = pd.crosstab(df['churn'], df['customer service calls'], margins=True)
# print(f"\ncrosstab churners customer service calls margins: \n {churner_serv_calls} \n")

df['many_service_calls'] = (df['customer service calls'] > 3).astype('int')
crosstab_servicecalls = pd.crosstab(df['many_service_calls'], df['churn'], margins=True)
print(f"\ncrosstab service calls > 3: \n { crosstab_servicecalls} \n")


# crosstab_servicecall_intlplan = pd.crosstab(df['many_service_calls'] & df['international plan'] , df['churn'], margins=True)
# print(f"\ncrosstab service calls & intl plan: \n {crosstab_servicecall_intlplan} \n")

# Thus we have used the condition that the users with less than 4 calls and international plan are likely to be loyal users

