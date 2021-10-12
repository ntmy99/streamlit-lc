import streamlit as st 
import numpy as np
import pandas 
from pandas import DataFrame
from plotly import express as px 
from plotly import validators
from scipy.stats import beta
import random
from cde_function import * 

st.set_page_config(page_title="Limit Control Dashboard", page_icon="", layout="wide")
html_header="""
<head>
<title>PControlDB</title>
<meta charset="utf-8">
<meta name="keywords" content="dashboard">
<meta name="description" content="limit control dashboard">
<meta name="viewport" content="width=device-width, initial-scale=1">
</head>
<h1 style="font-size:300%; color:#ffffff; font-family:Sans Serif"> LIMIT CONTROL DASHBOARD <br>
 <hr style= "  display: block;
  margin-top: 0.5em;
  margin-bottom: 0.5em;
  margin-left: auto;
  margin-left: auto;
  margin-right: auto;
  border-style: inset;
  border-width: 2px;"></h1>
"""
st.markdown(html_header, unsafe_allow_html=True)
st.markdown(""" <style>
#MainMenu {visibility: hidden;}
footer {visibility: hidden;}
</style> """, unsafe_allow_html=True)
st.sidebar.image: st.sidebar.image("logo-epay.svg", use_column_width=True)
st.sidebar.title("Dashboard Inputs")
st.markdown(
        """
Users must switch theme to Custom theme or Dark mode by clicking into 3-line button on the top right side, navigating to Settings and choose Custom/Dark mode under Theme.

Workflow of this dashboard: 

Step 1: Users submit required inputs (min loan, max loan, shape factor, number of loans, APR) to the sidebar before running.

Step 2: The program will find the optimal shape ratio that will generate highest return.

Step 3: Then it will simulate 1000 scenarios with optimised shape factor

Step 4: The final results will be printed out for evaluation. 

""")

#INPUTS####################################################################################
no_loans = st.sidebar.number_input("Number of Loans (<5000): ", min_value=1000, max_value=999999999, help = "The total number of loans that LC dashboard used expected in their simulation")
shape_factor = st.sidebar.number_input("Shape Factor(-0.08 to 0.08): ", min_value=-0.08, max_value=0.08, help = "Shape factor is a limit control parameters used to limit the loan amount")
min_loan = st.sidebar.number_input("Min Loan Amount: ", min_value=2000000, max_value=999999999, help = "The minimum loan amount in the simulation portfolio that LC dashboard user expected")
max_loan = st.sidebar.number_input("Max Loan Amount: ", min_value=5000000, max_value=999999999, help = "The maximum loan amount in the simulation portfolio that LC dashboard user expected" )
most_loan = st.sidebar.number_input("Mode Loan Amount: ", min_value=3500000, max_value=999999999,help = "The most frequent amount that customers usually request to borrow") 
APR = st.sidebar.number_input("APR: ", min_value=0.1, max_value=0.9, help = 'Annual Percentage Rate is an interest rate that is charged per customer per loan')
alpha_loan,alpha_pd = 5,2
beta_loan = (int(max_loan) - int(min_loan))*(int(alpha_loan) - 1)/(int(most_loan) - int(min_loan)) - int(alpha_loan) + 2 
beta_pd = ((0.99 - 0)*(int(alpha_pd)-1)/(0.2-0)) - int(alpha_pd) + 2
#INPUTS (which can be changed, might not be fixed):
LGD,no_payments,capital_ratio, funding_cost, decay_rate = 0.7,6,0.2,0.11,0.8
interest_rate = float(APR)/12
vn = (1+float(interest_rate))**(-float(no_payments))
an = (1-float(vn))/interest_rate
installment = 1/float(an)
#GENERATING DATA####################################################################################
loan_id, loan_amount, pd_list= fixed_simulation(no_loans,alpha_loan,beta_loan, min_loan, max_loan, alpha_pd, beta_pd)

w_list = [1]
for i in w_list: 
    if len(w_list)<6:
        weight = i*decay_rate
        w_list.append(weight)
cond_prob = [i/sum(w_list) for i in w_list]
acc_cond_p = cumul(cond_prob) #EXTRACT THIS TO USE
acc_nii, loan_loss = nii_table(interest_rate,installment,capital_ratio, funding_cost,decay_rate,APR,LGD,no_payments)

shape_f = [i if i !=-0.0 else 0.08 for i in [round(i,2) for i in np.arange(-0.08,0.08,0.01)]]

e_ncf, e_lcncf  = expected_ncf(simulation,loan_amount,pd_list,no_loans,acc_cond_p,loan_loss,min_loan,max_loan,acc_nii)
df_ncf = DataFrame(list(zip(shape_f,e_ncf)),columns = ['Parameters','ExpectedNCF'])
df_ncf.sort_values(by=['Parameters'], inplace=True)
df_lcncf = DataFrame(list(zip(shape_f,e_lcncf)),columns = ['Parameters','ExpectedNCF']) 
df_lcncf.sort_values(by=['Parameters'], inplace=True)

#PICKING THE BEST SHAPE FACTOR####################################################################################
shape_2 = max(dict(zip(shape_f,e_lcncf)), key=dict(zip(shape_f,e_lcncf)).get)
df_2,df_3 = simulation2(shape_2,loan_amount,pd_list,no_loans,acc_cond_p,loan_loss,min_loan,max_loan,acc_nii)

#VISUALIZING RESULTS####################################################################################
st.button("I. Data Analysis")
left_col, right_col = st.columns(2)
#1.LOAN 2D#################################################################################### 
df_loan = freq_table(loan_amount)
df_loan['Bin_range'] = df_loan['index'].str[11:19]
df_loan = df_loan[['Bin_range','Frequency']]
fig = px.bar(df_loan, x='Bin_range', y='Frequency')
fig.update_traces(marker_color='rgb(87,220,190)', marker_line_color='rgb(52,132,114)',
                  marker_line_width=0.5)
fig.update_xaxes(tickangle = -30)
fig_loan = update_chart(fig)
#2. PD 2D####################################################################################
df_pd = DataFrame(pd_list,columns = ['PD'])
fig1 = px.histogram(df_pd, x="PD", nbins=30)
fig1.update_traces(marker_color='rgb(87,220,190)', marker_line_color='rgb(52,132,114)',
                  marker_line_width=0.5 )
fig_pd = update_chart(fig1)

left_col.subheader('Loan Amount Histogram')  
left_col.plotly_chart(fig_loan,use_container_width=True)
right_col.subheader('PD Histogram')
right_col.plotly_chart(fig_pd,use_container_width=True)

st.button("II. Picking Shape Factor ")
#3. SHAPE FACTOR GRAPH ####################################################################################
pd = [i for i in np.arange(0,0.99,0.01)]
shape_df = lc_matrix2(pd,shape_factor,min_loan, max_loan)
fig_shape = go.Figure()
fig_shape.add_trace(go.Scatter(
    x=shape_df['PD'],
    y=shape_df['MaximumLoan'],
    mode="lines+text",
    line = dict(color='rgb(240, 149, 61)',width=4 )))
fig_shape.update_layout(title='Expected NCF',xaxis_title='Parameters',yaxis_title='NCF')
fig_shapef = update_chart(fig_shape)
html_hs="""
<h3 style="color:#ffffff; font-family:Sans Serif;">Limit Control Applied on Loan Amount</h3>
"""
st.markdown(html_hs, unsafe_allow_html=True)
st.plotly_chart(fig_shapef,use_container_width=True) 
#4. NCF 2D ####################################################################################
text_ncf = [str(i) for i in shape_f]
fig_ncf = ncf_2D(df_ncf,update_chart,text_ncf)
fig_lcncf = ncf_2D(df_lcncf,update_chart,text_ncf)

left_col1, right_col1 = st.columns(2)
left_col1.subheader('No Limit Control')  
left_col1.plotly_chart(fig_ncf,use_container_width=True)
right_col1.subheader('With Limit Control ')  
right_col1.plotly_chart(fig_lcncf,use_container_width=True) 

st.button("III. Simulating Results ")
#3D GRAPH (NCF) ##################################################################################################
fig5 = ncf_3d(df_2)
fig6 = ncf_3d(df_3)

#RESULT TABLE #####################################################################################################
col_name = ["",'No Limit Control','With Limit Control',"Portfolio Value",str("{:,}".format(sum(df_2['LoanAmount']))),str("{:,}".format(sum(df_3['LoanAmount']))),"Total NCF",str("{:,}".format(sum(df_2['NCF']))),str("{:,}".format(sum(df_3['NCF']))),'ROE',str(round(sum(df_2['NCF'])/(sum(df_2['LoanAmount'])*capital_ratio)*100,2))+ "%",str(round(sum(df_3['NCF'])/(sum(df_3['LoanAmount'])*capital_ratio)*100,2))+ "%"]
html_h3="""
<h3 style="color:#ffffff; font-family:Sans Serif;">One-time simulation result</h3>
"""
st.markdown(html_h3, unsafe_allow_html=True)
i_list = [0,3,6,9]
for i in i_list: 
    cols = st.columns(3)
    cols[0].write(col_name[i])
    cols[1].write(col_name[i+1])
    cols[2].write(col_name[i+2])
                        
shape_2 = max(dict(zip(shape_f,e_lcncf)), key=dict(zip(shape_f,e_lcncf)).get)
df_2,df_3 = simulation2(shape_2,loan_amount,pd_list,no_loans,acc_cond_p,loan_loss,min_loan,max_loan,acc_nii)

left_col2, right_col2 = st.columns(2)
left_col2.subheader("No Limit Control")
right_col2.subheader("With Limit Control")
left_col2.plotly_chart(fig5,use_container_width=True)
right_col2.plotly_chart(fig6,use_container_width=True) 
 
#SIMULATION WITH OPTIMISED SHAPE F ####################################################################################
optimised_list = [simulation(shape_2,loan_amount,pd_list,no_loans,acc_cond_p,loan_loss,min_loan,max_loan,acc_nii) for _ in range(1000)]
o_ncf,o_lcncf = list(zip(*optimised_list))

left_col3, right_col3 = st.columns(2)
df_oncf = final_odf(neg_label,pos_label, o_df, o_ncf, freq_table2)
fig7 = o_visualize(df_oncf,(120,117,203),(61, 58, 181))
fig7a = update_chart(fig7)
 
df_olcncf = final_odf(neg_label,pos_label, o_df, o_lcncf, freq_table2)
fig8 = o_visualize(df_olcncf,(120,117,203),(61, 58, 181))
fig8a = update_chart(fig8)

left_col3.subheader('NCF distribution (No LC)')  
left_col3.plotly_chart(fig7a,use_container_width=True)
right_col3.subheader('NCF distribution (With LC)')  
right_col3.plotly_chart(fig8a,use_container_width=True)

if sum(df_2['NCF'])< sum(df_3['NCF']) and sum(df_2['NCF']) > 0:
    decision = "RECOMMENDED"
    emoji = "white_check_mark"
else:
    decision = "NOT RECOMMENDED"
    emoji = "no_entry_sign"
html_subt1="""
<h2 style="color:#ffffff; font-family:Sans Serif;">Final Result</h2>
"""
st.markdown(html_subt1, unsafe_allow_html=True)

html_br="""
<br>
"""
st.markdown(html_br, unsafe_allow_html=True)

html_card_header6="""
<div class="card">
  <div class="card-body" style="border-radius: 10px 10px 0px 0px; background: #b4ceee; padding-top: 10px; width: 250px;
   height: 50px;">
    <h5 class="card-title" style="background-color:#b4ceee; color:#060f18; font-family:Georgia; text-align: center; padding: 5px 0;"> Dashboard Result</h5>
  </div>
</div>
"""
html_card_footer6="""
<div class="card">
  <div class="card-body" style="border-radius: 0px 0px 10px 10px; background: #b4ceee; padding-top: 1rem;; width: 250px;
   height: 50px;">
    <p class="card-title" style="background-color:#b4ceee; color:#060f18; font-family:Georgia; text-align: center; padding: 0px 0;"> </p>
  </div>
</div>
"""
html_card_header7="""
<div class="card">
  <div class="card-body" style="border-radius: 10px 10px 0px 0px; background: #b4ceee; padding-top: 5px; width: 250px;
   height: 50px;">
    <h5 class="card-title" style="background-color:#b4ceee; color:#060f18; font-family:Georgia; text-align: center; padding: 8px 0;">Recommendation</h5>
  </div>
</div>
"""
html_card_footer7="""
<div class="card">
  <div class="card-body" style="border-radius: 0px 0px 10px 10px; background: #b4ceee; padding-top: 1rem;; width: 250px;
   height: 50px;">
    <p class="card-title" style="background-color:#b4ceee; color:#060f18; font-family:Georgia; text-align: center; padding: 0px 0;"></p>
  </div>
</div>
"""

html_list="""
<ul style="color:#008080; font-family:Georgia; font-size: 15px">
</ul> 
"""

### Block 6#########################################################################################
with st.container():
    col1, col2, col3, col4, col5 = st.columns([2,12,1,12,2])
    with col1:
        st.write(" ")
    with col2:
        st.markdown(html_card_header6, unsafe_allow_html=True)
        st.markdown("For 1000 simulations:")
        st.markdown(f"**Total Portfolio Value:** {sum(df_3['LoanAmount']):,}")
        st.markdown(f"**Total NCF no LC:** {sum(o_ncf)/len(o_ncf):,}")
        st.markdown(f"**Total NCF with LC:** {sum(o_lcncf)/len(o_lcncf):,}")
        st.markdown(html_card_footer6, unsafe_allow_html=True)
    with col3:
        st.write("")
    with col4:
        st.markdown(html_card_header7, unsafe_allow_html=True)
        st.markdown(f"**Final decision:** {decision} :{emoji}:")
        st.markdown("**For this portfolio...**")
        st.markdown(f"**APR used is:** {float(APR):,}")
        st.markdown(f"**Shape Factor used is:** {float(shape_2):,}")
        st.markdown(html_card_footer7, unsafe_allow_html=True)

html_line="""
<br>
<br>
<hr style= "  display: block;
  margin-top: 0.5em;
  margin-bottom: 0.5em;
  margin-left: auto;
  margin-right: auto;
  border-style: inset;
  border-width: 2px;">
<p style="color:Gainsboro; text-align: right;"></p>
"""
st.markdown(html_line, unsafe_allow_html=True)
c1,c2 = st.columns(2)
c2.image('logo-epay.svg', use_column_width=True)
