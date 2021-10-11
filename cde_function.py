import streamlit as st 
import numpy as np
import pandas as pd
from pandas import DataFrame
from plotly import express as px 
from plotly import graph_objects as go
from plotly import subplots
from plotly import validators
from scipy.stats import beta
import random
#FIXED SIMULATION 
def fixed_simulation(no_loans,alpha_loan,beta_loan, min_loan, max_loan, alpha_pd, beta_pd): 
    loan_id = [i+1 for i in range(0,int(no_loans),1)]
    loan_amount = [round(beta.ppf(random.randint(0, 100) / 100, int(alpha_loan),
             int(beta_loan), loc=int(min_loan), scale=int(max_loan)
             - int(min_loan))) for i in range(int(no_loans))]
    pd_list = [round(beta.ppf(random.randint(0, 100) / 100, float(alpha_pd),
               float(beta_pd), loc=0, scale=0.99),2) for i in
               range(int(no_loans))]
    return loan_id, loan_amount, pd_list 
#PARAMETERS(NII, LOANLOSS VECTOR)
def cumul(a):
        total  = 0
        sums   = []
        for v in a:
            total = total + v
            sums.append(total)
        return sums 
    
def loan_schedule(list_prev,capital_ratio,interest_rate, installment, funding_cost,LGD):
    list_new = []    
    e = round(list_prev[0] - list_prev[2],2)
    wof = e*interest_rate
    p = installment - wof
    c = round(e - p,2)
    equity = e*capital_ratio
    fund = e - equity
    cost_fund = fund*funding_cost/12
    loan_loss = e*LGD
    list_new.extend((e,wof,p,c,equity,fund,loan_loss,cost_fund))
    return list_new 

def nii_table(interest_rate,installment,capital_ratio, funding_cost,decay_rate,APR,LGD,no_payments):    
    #USE LOAN LOSS AND ACC_NII 
    #inputs required: decayrate, APR, LGD, no_payments, interest_rate, installment, funding cost
    period_1 = [1]
    for i in period_1: 
        wof= i*interest_rate
        p = installment - wof
        c = i - p
        equity = i*capital_ratio 
        fund = i - equity 
        cost_fund = fund*funding_cost/12
        loan_loss = i*LGD
        period_1.extend((wof,p,c,equity,fund,loan_loss,cost_fund))
        break
        
    a = [i+1 for i in range(0,7,1)]
    new_list = [period_1]
    for i in range(len(a)):
        if i >=1:
            e = loan_schedule(new_list[i-1],capital_ratio,interest_rate, installment, funding_cost,LGD)
            new_list.append(e)
    loan_df = pd.DataFrame(new_list,columns = ['EAD','WOF','Principal','Closing','Equity','Funding Balance','Loan Loss','Cost Fund'])

    wof,cof = loan_df['WOF'].to_list(),loan_df['Cost Fund'].to_list() 
    wof_sum = cumul(wof)

    sum_cf = [cof[0] if i==0 else cof[i -1]+cof[i] for i in range(len(cof))]
    e1 = sum_cf[0]
    nii_list = [-e1]    
    for i,v1 in enumerate(sum_cf):
        if i>0: 
            nii = wof_sum[i-1] - v1   
            nii_list.append(nii)

    a,val2 = nii_list[0],2*nii_list[0]
    acc_nii = []
    for index,value in enumerate(nii_list):
        if index <= 1:
            val1 = (index+1)*nii_list[0]
            acc_nii.append(val1)
        else: 
            val2 = val2 + nii_list[index-1]
            acc_nii.append(val2)
    return acc_nii, loan_df['Loan Loss'].to_list()
#Limit Control Matrix
def lc_matrix(pd,shape_factor,min_loan, max_loan):
        alpha = 1 + shape_factor
        vn1 = alpha**(-99)
        an1 = (1-vn1)/shape_factor
        beta = 1/an1
        alpha_list = [alpha**(100*i)  for i in pd]
        scale = [value - beta*(value-1)/shape_factor if index !=98 else round((value - beta*(value-1)/shape_factor),1) for index,value in enumerate(alpha_list)] 
        maximum_loan = [round(min_loan + i*(max_loan-min_loan)) for i in scale]
        return maximum_loan    
def simulation(shape_factor,loan_amount,pd_list,no_loans,acc_cond_p,loan_loss,min_loan,max_loan,acc_nii):
    #Simulation from here: 
    default = [1 if v > random.randint(0, 100)/100 else 0 for v in pd_list]
    prob_rand = [round(random.randint(0, 100)/100 + 0.005,2) for i in range(int(no_loans))]
    result = [7 if value ==0 else random.randint(0, 100)/100 for value in default]
    times_at_default = []
    for value in result:
            if value ==7: 
                e = value 
            else: 
                if value <= acc_cond_p[0]:
                    e = 1
                if acc_cond_p[0] < value <= acc_cond_p[1]:
                    e = 2
                if acc_cond_p[1] < value <= acc_cond_p[2]:
                    e = 3
                if acc_cond_p[2] < value <= acc_cond_p[3]:
                    e = 4
                if acc_cond_p[3] < value <= acc_cond_p[4]:
                    e = 5
                if acc_cond_p[4] < value <= acc_cond_p[5]:
                    e = 6 
            times_at_default.append(e)

    abc = [0 if value == 7 else loan_loss[value-1] for value in times_at_default]
    loanloss_amt = [round(v1*v2,2) for v1, v2 in list(zip(abc,loan_amount))]
    #applied NII: 
    maximum_loan = lc_matrix(pd_list,shape_factor,min_loan, max_loan) 
    lc_loan = []
    for v1,v2 in zip(maximum_loan,loan_amount):
        if v1>= min_loan and v1<max_loan:
            value = min(v1,v2)
            lc_loan.append(value) 
        elif v1 < min_loan:
            lc_loan.append(min_loan)
        elif v1 == max_loan: 
            lc_loan.append(v2)
    lc_loanloss = [round(v1*v2,2) for v1, v2 in zip(abc,lc_loan)]        
    lookup_p = [acc_nii[v1 - 1] for v1 in times_at_default]
    nii_list = [round(v1*v2) for v1, v2 in list(zip(lookup_p,loan_amount))]                
    lc_nii = [round(v1*v2) for v1, v2 in list(zip(lookup_p,lc_loan))]                
    ncf = [round(v1 - v2) for v1, v2 in list(zip(nii_list,loanloss_amt))]
    lc_ncf = [round(v1 - v2) for v1, v2 in list(zip(lc_nii,lc_loanloss))]
    return (sum(ncf),sum(lc_ncf))
def simulation2(shape_factor,loan_amount,pd_list,no_loans,acc_cond_p,loan_loss,min_loan,max_loan,acc_nii):
    #Simulation from here: 
    default = [1 if v > random.randint(0, 100)/100 else 0 for v in pd_list]
    prob_rand = [round(random.randint(0, 100)/100 + 0.005,2) for i in range(int(no_loans))]
    result = [7 if value ==0 else random.randint(0, 100)/100 for value in default]
    times_at_default = []
    for value in result:
            if value ==7: 
                e = value 
            else: 
                if value <= acc_cond_p[0]:
                    e = 1
                if acc_cond_p[0] < value <= acc_cond_p[1]:
                    e = 2
                if acc_cond_p[1] < value <= acc_cond_p[2]:
                    e = 3
                if acc_cond_p[2] < value <= acc_cond_p[3]:
                    e = 4
                if acc_cond_p[3] < value <= acc_cond_p[4]:
                    e = 5
                if acc_cond_p[4] < value <= acc_cond_p[5]:
                    e = 6 
            times_at_default.append(e)

    abc = [0 if value == 7 else loan_loss[value-1] for value in times_at_default]
    loanloss_amt = [round(v1*v2,2) for v1, v2 in list(zip(abc,loan_amount))]
    #applied NII: 
    maximum_loan = lc_matrix(pd_list,shape_factor,min_loan, max_loan) 
    lc_loan = []
    for v1,v2 in zip(maximum_loan,loan_amount):
        if v1>= min_loan and v1<max_loan:
            value = min(v1,v2)
            lc_loan.append(value) 
        elif v1 < min_loan:
            lc_loan.append(min_loan)
        elif v1 == max_loan: 
            lc_loan.append(v2)
    lc_loanloss = [round(v1*v2,2) for v1, v2 in zip(abc,lc_loan)]        
    lookup_p = [acc_nii[v1 - 1] for v1 in times_at_default]
    nii_list = [round(v1*v2) for v1, v2 in list(zip(lookup_p,loan_amount))]                
    lc_nii = [round(v1*v2) for v1, v2 in list(zip(lookup_p,lc_loan))]                
    ncf = [round(v1 - v2) for v1, v2 in list(zip(nii_list,loanloss_amt))]
    lc_ncf = [round(v1 - v2) for v1, v2 in list(zip(lc_nii,lc_loanloss))]
    df_2 = DataFrame(list(zip(loan_amount,pd_list,ncf)),columns = ['LoanAmount','PD','NCF'])
    df_3 = DataFrame(list(zip(lc_loan,pd_list,lc_ncf)),columns = ['LoanAmount','PD','NCF'])                   
    return df_2,df_3 
def expected_ncf(function,loan_amount,pd_list,no_loans,acc_cond_p,loan_loss,min_loan,max_loan,acc_nii): 
    shape_f = [i if i != -0.0 else 0.08 for i in [round(i,2) for i in np.arange(-0.08,0.08,0.01)]]
    ncf_list = []
    lcncf_list = [] 
    for i in shape_f: 
        a = [function(i,loan_amount,pd_list,no_loans,acc_cond_p,loan_loss,min_loan,max_loan,acc_nii) for x in range(100)]
        ncf_l,lcncf_l = list(zip(*a))
        ncf_list.append(sum(ncf_l)/100)
        lcncf_list.append(sum(lcncf_l)/100)
    return ncf_list, lcncf_list 
def freq_table(col):
    df = DataFrame(col,columns = ['Col'])
    df['Frequency'] = pd.cut(df['Col'], 25).astype(str)
    df.sort_values(by=['Col'],inplace = True,ignore_index = True)
    df2 = df['Frequency'].value_counts().sort_index(ascending=True).to_frame().reset_index()
    return df2
def lc_matrix2(pd,shape_factor,min_loan, max_loan):
    alpha = 1 + shape_factor
    vn1 = alpha**(-99)
    an1 = (1-vn1)/shape_factor
    beta = 1/an1
    alpha_list = [alpha**(100*i)  for i in pd]
    scale = [value - beta*(value-1)/shape_factor if index !=98 else round((value - beta*(value-1)/shape_factor),1) for index,value in enumerate(alpha_list)] 
    maximum_loan = [round(min_loan + i*(max_loan-min_loan)) for i in scale]
    a = list(zip(pd,maximum_loan))
    df_1 = DataFrame(a,columns = ['PD','MaximumLoan'])
    return df_1 
def update_chart(fig): 
    fig.update_layout(
        xaxis=dict(
            showline=True,
            showgrid=False,
            zeroline=False
        ),
        yaxis=dict(
            showgrid=False,
        ),
        autosize=False,
        margin=dict(
            autoexpand=False,
            l=100,
            r=20,
            t=110,
        ),
    )
    fig.update_layout(hovermode='x unified')
    return fig 
def ncf_2D(df,update_chart,text_ncf):
    fig = go.Figure()
    fig.add_trace(go.Scatter(
    x=df['Parameters'],
    y=df['ExpectedNCF'],
    mode="lines+markers+text",
    text=text_ncf,
    textposition="top center",
    line = dict(color='royalblue',width=4)))
    fig.update_layout(title='Expected NCF',xaxis_title='Parameters',yaxis_title='NCF')
    fig_ncf = update_chart(fig)
    return fig_ncf
def ncf_3d(df):
    fig = go.Figure(data =[go.Scatter3d(x = df['LoanAmount'],
                                    y = df['PD'],
                                    z = df['NCF'],
                                    mode ='markers', 
                                    marker = dict(
                                        size = 3,
                                        color = df['NCF'],
                                        colorscale ='rainbow',
                                        opacity = 0.8
                                    )
    )])
    fig.update_layout(scene = dict(
                        xaxis_title='Loan Amount',
                        yaxis_title='PD',
                        zaxis_title='NCF'),
                        title="Net Cash Flow",
                        font=dict(
                            family="Sans Serif",
                            size=12
                        )
    )
    fig.update_layout(template='plotly_dark')
    return fig 

def freq_table2(col,label):
    df = DataFrame(col,columns = ['Col'])
    df['Frequency'] = pd.cut(df['Col'], 30,labels = label)
    df.sort_values(by=['Frequency'],inplace = True,ignore_index = True)
    df2 = df['Frequency'].value_counts().sort_index(ascending=True).to_frame().reset_index()
    return df2

def o_df(list1,function,label):
    o_ncf2 = [int(i/1000000) for i in list1]
    df_result1 = function(o_ncf2,label) 
    df_result1['Bin'] = df_result1['index'].astype(str) 
    df_result11 = df_result1[['Bin','Frequency']]
    return df_result11 

##TH positive: (1) if positive then bin will be 7 to 99, 99 to 200, and 200-400
##TH negative: (1) if negative then bin will be -7 to 99,99 to 200, and 200-400
def neg_label(o_df, name_list,function,label):
    a = o_df(name_list,function,label)
    bin_int = []
    for i,v in enumerate(a['Bin'].to_list()):
        try:
            try:
                        try:
                            bin_int.append(int(v[1:5]))
                        except:
                            bin_int.append(int(v[1:4]))
            except: 
                bin_int.append(int(v[1:3]))
        except: 
            bin_int.append(int(v[1:2]))
    index_bin = [] 
    if len(str(bin_int[0])) >= 2 and len(str(bin_int[1]))>=2 and  len(str(bin_int[0])) <=3: 
        index_bin = [i for i,v in enumerate(bin_int) if v in range(-100,-89,1)]
    elif len(str(bin_int[0]))>=4: 
        index_bin = [i for i,v in enumerate(bin_int) if v in range(-1000,-890,1)]

    new_bin = []
    for i,v in enumerate(a['Bin'].to_list()):
            if bool(index_bin) == True and i<=max(index_bin):
                try:
                    new_bin.append(int(v[1:5]))
                except: 
                    new_bin.append(int(v[1:4]))
            elif bool(index_bin) == True and i>max(index_bin):
                new_bin.append(int(v[1:4]))
            elif bool(index_bin) == False:
                new_bin = [int(v) for v in bin_int]
    return new_bin 

def pos_label(o_df, name_list,function,label):
    a = o_df(name_list,function,label)
    bin_int = []
    for i,v in enumerate(a['Bin'].to_list()):
        try:
            try:
                        try:
                            bin_int.append(int(v[1:5]))
                        except:
                            bin_int.append(int(v[1:4]))
            except: 
                bin_int.append(int(v[1:3]))
        except:
            bin_int.append(int(v[1:2]))
    index_bin = [] 
    if len(str(bin_int[0])) >= 2 and len(str(bin_int[0])) <=3: 
        index_bin = [i for i,v in enumerate(bin_int) if v in range(-100,-89,1)]
    elif len(str(bin_int[0]))>=4: 
        index_bin = [i for i,v in enumerate(bin_int) if v in range(-1000,-890,1)]

    new_bin = []
    for i,v in enumerate(a['Bin'].to_list()):
            if bool(index_bin) == True and i<=max(index_bin):
                try:
                    new_bin.append(int(v[1:5]))
                except: 
                    new_bin.append(int(v[1:4]))
            elif bool(index_bin) == True and i>max(index_bin):
                new_bin.append(int(v[1:4]))
            elif bool(index_bin) == False:
                new_bin = [int(v) for v in bin_int]
    return new_bin


#These functions are used to change the label of bin range
#use freq_table2 two times, first time is to find the interval (labels = none), second time is to embedded the new label into d
def final_odf(neg_label,pos_label, o_df, name_list, freq_table2):
    if sum(name_list) <=0: 
        labels_ncf = neg_label(o_df,name_list,freq_table2,None)
    else:    
        labels_ncf = pos_label(o_df,name_list,freq_table2,None)
    df_oncf = o_df(name_list,freq_table2,labels_ncf)
    df_oncf['Bin'] = df_oncf['Bin'].astype('int64') 
    return df_oncf

def o_visualize(df,a,b):
    fig = px.bar(df, x='Bin', y='Frequency')
    fig.update_traces(marker_color='rgb' + str(a), marker_line_color='rgb' + str(b),
                      marker_line_width=1.5)
    fig.update_xaxes(tickangle = -30)
    return fig
