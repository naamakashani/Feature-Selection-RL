# -*- coding: utf-8 -*-
"""
Created on Tue Sep 21 17:25:47 2021

@author: dafna
"""


# -*- coding: utf-8 -*-
"""
Created on Thu Apr 16 13:39:46 2020
@author: urixs
"""

import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import CountVectorizer
from sklearn import preprocessing
from preprocess_ct_re import *
import datetime
from datetime import datetime
from collections import Counter

datetime_format = "%d/%m/%Y %H:%M"
MAX_DATE = '2021-04-20'
TRAIN_MAX_DATE = '2017-12-31'
VAL_MAX_DATE = '2018-12-31'
DEFAULT_TEXT_MAX_SENTENCES = 10


filenames = {
    'data/Peumonia - 23.10.21.xlsx':'Pneumonia',
    'data/ACS 23.10.21.xlsx': 'ACS',
    # 'data/ICHfinilizing2 -22.10.21.xlsx': 'ICHfinilizing',
    'data/ischemicstroke - 22.10.21.xlsx' : 'Ischemicstroke',
    'data/appendicitis - 21.10.21.xlsx': 'Appendicitis',
    'data/Headoffemur - 22.4.21 (1).xlsx' : 'Headoffemur'
    
    #---------------------------------------
    
    # 'data/appendicitis - 21.10.21.xlsx'
    # 'data/ischemicstroke - 16.4.21.xlsx',
    # 'data/ICHfinilizing2 - 6.4.21.xlsx',
    # 'data/Headoffemur - 22.4.21.xlsx',
    # 'data/appendicitis - 22.4.21.xlsx'
    
    }
# Create dataframes for explanatory variables and outcome variables
outcomes = pd.DataFrame()
X = pd.DataFrame()


def get_all_files_features_and_outcomes(filenames):
    dtd_le_30 = pd.Series()
    dtd_le_60 = pd.Series() 
    dtd_le_90 =pd.Series()
    dtd_le_120 = pd.Series()
    readmission_in_30_days = pd.Series()
    readmission_in_60_days = pd.Series()
    readmission_in_90_days= pd.Series()
    readmission_in_120_days= pd.Series()
    days_to_first_readmission= pd.Series()
    days_to_death= pd.Series()
    admission_date= pd.Series()
    partition= pd.Series()
    readmission_class= pd.Series()
    gender= pd.Series()
    age= pd.Series()
    esi= pd.Series() 
    ct_exam_order_creation_visit_start= pd.Series()
    is_past_or_present_smoker= pd.Series()
    is_present_smoker= pd.Series()
    
    cbc_from_visit_start= pd.Series()
    cbc_plt_collection_date_hrs= pd.Series()
    is_hyper_tension= pd.Series()
    is_diabetic= pd.Series()
    is_congestive_heart_failure= pd.Series()
    ct_exam_order_creation_exam_start= pd.Series()
    ct_done= pd.Series() 
    ct_exam_performed= pd.Series() 
    medications_anticoagulants= pd.Series() 
    medications_hypertnesive= pd.Series() 
    diagnosis = pd.Series()
    ill_type = pd.Series()
    exam_performed_type = pd.Series()
    exam_performed_subtype = pd.Series()
    exam_performed_res = pd.Series() 
    ct_exam_res = pd.Series()
    us_exam_res = pd.Series()
    xray_exam_res = pd.Series()
    all_exam_performed_res = pd.Series()
    
    quantile_wbc= pd.Series() 
    quantile_hgb= pd.Series()
    quantile_plt= pd.Series()
    quantile_temperature = pd.Series()
    quantile_pulse = pd.Series()
    quantile_BP_SYS = pd.Series()
    quantile_BP_DIAS = pd.Series()
    quantile_CRP = pd.Series()
    quantile_troponin = pd.Series()
    
    
    
    
    for filename in filenames.keys():
        i_dtd_le_30, i_dtd_le_60, i_dtd_le_90, i_dtd_le_120,\
        i_readmission_in_30_days, i_readmission_in_60_days, i_readmission_in_90_days, i_readmission_in_120_days,\
        i_days_to_first_readmission,i_days_to_death,i_admission_date,i_partition,i_readmission_class,\
        i_gender,i_age,i_esi, i_ct_exam_order_creation_visit_start,i_is_past_or_present_smoker,\
        i_is_present_smoker,i_quantile_wbc, i_quantile_hgb,i_quantile_plt,\
        i_cbc_from_visit_start,i_cbc_plt_collection_date_hrs,i_is_hyper_tension,i_is_diabetic,\
        i_is_congestive_heart_failure,\
        i_ct_exam_order_creation_exam_start,i_ct_done, \
        i_ct_exam_performed, i_medications_anticoagulants, i_medications_hypertnesive, i_diagnosis, \
        i_exam_performed_type, i_exam_performed_subtype, i_exam_performed_res,\
        i_quantile_temperature, i_quantile_pulse, i_quantile_BP_SYS,i_quantile_BP_DIAS,\
        i_quantile_CRP, i_quantile_troponin, \
        i_ct_exam_res, i_us_exam_res, i_xray_exam_res, i_all_exam_performed_res = get_features_and_outcomes(filename)
        
        dtd_le_30 = pd.concat([dtd_le_30, i_dtd_le_30], ignore_index=True)
        dtd_le_60 = pd.concat([dtd_le_60, i_dtd_le_60], ignore_index=True)
        dtd_le_90 = pd.concat([dtd_le_90, i_dtd_le_90], ignore_index=True)
        dtd_le_120 = pd.concat([dtd_le_120, i_dtd_le_120], ignore_index=True)
        readmission_in_30_days = pd.concat([readmission_in_30_days, i_readmission_in_30_days], ignore_index=True)
        readmission_in_60_days = pd.concat([readmission_in_60_days, i_readmission_in_60_days], ignore_index=True)
        readmission_in_90_days = pd.concat([readmission_in_90_days, i_readmission_in_90_days], ignore_index=True)
        readmission_in_120_days= pd.concat([readmission_in_120_days, i_readmission_in_120_days], ignore_index=True)
        days_to_first_readmission= pd.concat([days_to_first_readmission, i_days_to_first_readmission], ignore_index=True)
        days_to_death= pd.concat([days_to_death, i_days_to_death], ignore_index=True)
        admission_date= pd.concat([admission_date, i_admission_date], ignore_index=True)
        partition= pd.concat([partition, i_partition], ignore_index=True)
        readmission_class= pd.concat([readmission_class, i_readmission_class], ignore_index=True)
        gender= pd.concat([gender, i_gender], ignore_index=True)
        age= pd.concat([age, i_age], ignore_index=True)
        esi= pd.concat([esi, i_esi], ignore_index=True)
        ct_exam_order_creation_visit_start= pd.concat([ct_exam_order_creation_visit_start, i_ct_exam_order_creation_visit_start], ignore_index=True)
        is_past_or_present_smoker= pd.concat([is_past_or_present_smoker, i_is_past_or_present_smoker], ignore_index=True)
        is_present_smoker= pd.concat([is_present_smoker, i_is_present_smoker], ignore_index=True)
        quantile_wbc= pd.concat([quantile_wbc, i_quantile_wbc], ignore_index=True)
        quantile_hgb= pd.concat([quantile_hgb, i_quantile_hgb], ignore_index=True)
        quantile_plt= pd.concat([quantile_plt, i_quantile_plt], ignore_index=True)
        cbc_from_visit_start= pd.concat([cbc_from_visit_start, i_cbc_from_visit_start], ignore_index=True)
        cbc_plt_collection_date_hrs= pd.concat([cbc_plt_collection_date_hrs, i_cbc_plt_collection_date_hrs], ignore_index=True)
        is_hyper_tension= pd.concat([is_hyper_tension, i_is_hyper_tension], ignore_index=True)
        is_diabetic= pd.concat([is_diabetic, i_is_diabetic], ignore_index=True)
        is_congestive_heart_failure= pd.concat([is_congestive_heart_failure, i_is_congestive_heart_failure], ignore_index=True)
        ct_exam_order_creation_exam_start= pd.concat([ct_exam_order_creation_exam_start, i_ct_exam_order_creation_exam_start], ignore_index=True)
        ct_done= pd.concat([ct_done, i_ct_done], ignore_index=True)
        ct_exam_performed= pd.concat([ct_exam_performed, i_ct_exam_performed], ignore_index=True)
        medications_anticoagulants= pd.concat([medications_anticoagulants, i_medications_anticoagulants], ignore_index=True)
        medications_hypertnesive= pd.concat([medications_hypertnesive, i_medications_hypertnesive], ignore_index=True)
        diagnosis = pd.concat([diagnosis, i_diagnosis], ignore_index=True)
        
        ill_type = pd.concat([ill_type, pd.Series([filenames[filename]] * len(i_diagnosis))], ignore_index=True)
        # ill_type = pd.concat([ill_type, pd.Series([''] * len(i_diagnosis))])
        
        exam_performed_type = pd.concat([exam_performed_type, i_exam_performed_type], ignore_index=True)
        exam_performed_subtype = pd.concat([exam_performed_subtype, i_exam_performed_subtype], ignore_index=True)
        exam_performed_res = pd.concat([exam_performed_res, i_exam_performed_res], ignore_index=True)
        
        quantile_temperature = pd.concat([quantile_temperature, i_quantile_temperature], ignore_index=True)
        quantile_pulse = pd.concat([quantile_pulse, i_quantile_pulse], ignore_index=True)
        quantile_BP_SYS = pd.concat([quantile_BP_SYS, i_quantile_BP_SYS], ignore_index=True)
        quantile_BP_DIAS = pd.concat([quantile_BP_DIAS, i_quantile_BP_DIAS], ignore_index=True)
        quantile_CRP = pd.concat([quantile_CRP, i_quantile_CRP], ignore_index=True)
        quantile_troponin = pd.concat([quantile_troponin, i_quantile_troponin], ignore_index=True)
        
        ct_exam_res = pd.concat([ct_exam_res, i_ct_exam_res], ignore_index=True)
        us_exam_res = pd.concat([us_exam_res, i_us_exam_res], ignore_index=True)
        xray_exam_res = pd.concat([xray_exam_res, i_xray_exam_res], ignore_index=True)
        all_exam_performed_res = pd.concat([all_exam_performed_res, i_all_exam_performed_res], ignore_index=True)

    return dtd_le_30, dtd_le_60, dtd_le_90, dtd_le_120,\
            readmission_in_30_days, readmission_in_60_days, readmission_in_90_days, readmission_in_120_days,\
            days_to_first_readmission,days_to_death,admission_date,partition,readmission_class,\
            gender,age,esi, ct_exam_order_creation_visit_start,is_past_or_present_smoker,\
            is_present_smoker,quantile_wbc, quantile_hgb,quantile_plt,\
            cbc_from_visit_start,cbc_plt_collection_date_hrs,is_hyper_tension,is_diabetic,\
            is_congestive_heart_failure,\
            ct_exam_order_creation_exam_start,ct_done, \
            ct_exam_performed, medications_anticoagulants, medications_hypertnesive, diagnosis, \
            ill_type, exam_performed_type, exam_performed_subtype, exam_performed_res,\
            quantile_temperature, quantile_pulse, quantile_BP_SYS,quantile_BP_DIAS,\
            quantile_CRP, quantile_troponin, \
            ct_exam_res, us_exam_res, xray_exam_res, all_exam_performed_res
            


def get_features_and_outcomes(filename):
    
    Data = pd.read_excel(filename)
    Data.columns= Data.columns.str.lower()


    # np.datetime64(MAX_DATE)
    #prev-format Data = Data.loc[pd.to_datetime(Data['Reference Event-Visit Start Date'],format=datetime_format) < datetime.datetime(2021, 4,4 )]
    Data = Data.loc[pd.to_datetime(Data['reference event-date of documentation'],format=datetime_format) < datetime.fromisoformat(MAX_DATE)]
    
    
    n = len(Data)
    Data = Data.reset_index()

    
    


    #################### Outcome variables ####################

    # death event indicator
    is_dead = Data['date of death'].notnull()
    
    # number of days from start of first visit to death
    #prev-format days_to_death = pd.to_datetime(Data['Date of Death'],format=datetime_format) - \
    #prev-format                 pd.to_datetime(Data['Reference Event-Visit Start Date'],format=datetime_format)
    
    days_to_death = pd.to_datetime(Data['date of death'],format=datetime_format) - \
                    pd.to_datetime(Data['reference event-date of documentation'],format=datetime_format)
    
    for i in range(n):
        if days_to_death.notnull()[i]:
            days_to_death[i] = days_to_death[i].days 

        
    days_to_death[days_to_death.isnull()] = None
    days_to_death = days_to_death.astype('float32')
    
    
    # whether the patient died less than 120 days after - use this as outcome.
    dtd_le_30 = (days_to_death <= 30) * 1.
    dtd_le_60 = (days_to_death <= 60) * 1.
    dtd_le_90 = (days_to_death <= 90) * 1.
    dtd_le_120 = (days_to_death <= 120) * 1.

    # 0: 120 < dtd, 1: 90< dtd < 120 2: 90 < dtd < 60, 3: 60 < dtd < 30, 4: dtd < 30
    dtd_class = dtd_le_120 + dtd_le_90 + dtd_le_60 + dtd_le_30
    
    # diagnosis
    #prev-format diagnosis_org = Data['Reference Event-Diagnoses'].copy()
    diagnosis = Data['reference event-diagnosis'].copy()
    # diagnosis_org = Data['Reference Event-Diagnosis'].copy()
    # diagnosis_org_distinct_vals = diagnosis_org.value_counts()
    # diagnosis_total_count = diagnosis_org_distinct_vals.sum()
    # for idiag in range(len(diagnosis_org_distinct_vals)):
    #     if (diagnosis_org_distinct_vals[idiag]/diagnosis_total_count < 0.1):
    #         diagnosis_org.loc[diagnosis_org == diagnosis_org_distinct_vals.index[idiag]] = "Too low frequency diagnosis"

    # diagnosis = pd.get_dummies(diagnosis_org.astype('category')) 

    # tia = np.zeros(n)
    # for i in range(n):
    #     if ((str)(diagnosis_org[i]).startswith('TIA')):
    #         tia[i] = 1
    #     if ("transient cerebral ischemia" in (str)(diagnosis_org[i]).lower()):
    #         tia[i] = 1
    # tia = pd.Series(tia)





    # Readmission counts
    #prev-format days_to_first_readmission =  pd.to_datetime(Data['readmit_1 2 yrs-Hospital Arrival Date'], format= datetime_format) - \
    
    #prev-format                              pd.to_datetime(Data['Reference Event-Visit Start Date'], format= datetime_format)
    
    
    # days_to_first_readmission =  pd.to_datetime(Data['readmit_1 2 yrs-Hospital Arrival Date'], format= datetime_format) - \
    #                              pd.to_datetime(Data['Reference Event-Date of Documentation'], format= datetime_format)
    days_to_first_readmission = []
    if 'readmit_1 2 yrs-hospital admission date-days from reference' in Data.columns:
        days_to_first_readmission = Data['readmit_1 2 yrs-hospital admission date-days from reference']
        for i in range(n):
            if not pd.isnull(days_to_first_readmission[i]):
                if days_to_first_readmission[i] < 1:
                    days_to_first_readmission[i] =Data['readmit_2 2 yrs-hospital admission date-days from reference'][i]
                    
                # days_to_first_readmission[i] = days_to_first_readmission[i].days 
        for i in range(n):
            if not pd.isnull(days_to_first_readmission[i]):
                if days_to_first_readmission[i] < 1:
                    days_to_first_readmission[i] =Data['readmit_3 2 yrs-hospital admission date-days from reference'][i]
    
    else:
        days_to_first_readmission =  pd.to_datetime(Data['readmit_1 120 d-hospital admission date'], format= datetime_format) - \
                                  pd.to_datetime(Data['reference event-date of documentation'], format= datetime_format)
        days_to_2nd_readmission = pd.to_datetime(Data['readmit_2 120 d-hospital admission date'], format= datetime_format) - \
                                  pd.to_datetime(Data['reference event-date of documentation'], format= datetime_format)
        days_to_3rd_readmission = pd.to_datetime(Data['readmit_3 120 d-hospital admission date'], format= datetime_format) - \
                                  pd.to_datetime(Data['reference event-date of documentation'], format= datetime_format)
                                  
        for i in range(n):
            if not pd.isnull(days_to_first_readmission[i]):
                days_to_first_readmission[i] = days_to_first_readmission[i].days
            if not pd.isnull(days_to_2nd_readmission[i]):
                days_to_2nd_readmission[i] = days_to_2nd_readmission[i].days
            if not pd.isnull(days_to_3rd_readmission[i]):
                days_to_3rd_readmission[i] = days_to_3rd_readmission[i].days
                
        for i in range(n):
            if not pd.isnull(days_to_first_readmission[i]):
                if days_to_first_readmission[i] < 1:
                    days_to_first_readmission[i] = days_to_2nd_readmission[i]
                    
                # days_to_first_readmission[i] = days_to_first_readmission[i].days 
        for i in range(n):
            if not pd.isnull(days_to_first_readmission[i]):
                if days_to_first_readmission[i] < 1:
                    days_to_first_readmission[i] =days_to_3rd_readmission[i]

    

    days_to_death[days_to_death.isnull()] = None
    days_to_death = days_to_death.astype('float32') 
    days_to_first_readmission[days_to_first_readmission.isnull()] = None
    days_to_first_readmission = days_to_first_readmission.astype('float32')        
            
    first_readmission_within_30_days = days_to_first_readmission <= 30 
    first_readmission_within_60_days = days_to_first_readmission <= 60 
    first_readmission_within_90_days = days_to_first_readmission <= 90 
    first_readmission_within_120_days = days_to_first_readmission <= 120 


        
    readmission_in_30_days = first_readmission_within_30_days * 1 #+ \
                              #second_readmission_within_30_days * 1 + \
                              #third_readmission_within_30_days * 1  
                              
    readmission_in_60_days = first_readmission_within_60_days * 1 #+ \
                              #second_readmission_within_60_days * 1 + \
                              #third_readmission_within_60_days * 1 
    
    readmission_in_90_days = first_readmission_within_90_days * 1 #+ \
                              #second_readmission_within_90_days * 1 + \
                              #third_readmission_within_90_days * 1 
    
    readmission_in_120_days = first_readmission_within_120_days * 1 #+ \
                               #second_readmission_within_120_days * 1 + \
                               #third_readmission_within_120_days * 1                              
    
    # 0: 120 < readmission, 
    # 1: 90< readmission < 120 ,
    # 2: 90 < readmission < 60, 
    # 3: 60 < readmission < 30, 
    # 4: readmission < 30
    readmission_class = first_readmission_within_120_days * 1 + \
                        first_readmission_within_90_days * 1 + \
                        first_readmission_within_60_days * 1 + \
                        first_readmission_within_30_days * 1
                        
    # admission time (for train / val / test partition in survival curves)                    
    #prev-format admission_date = [x.date() for x in pd.to_datetime(Data['Reference Event-Visit Start Date'], format = datetime_format)]
    admission_date = pd.Series([x.date() for x in pd.to_datetime(Data['reference event-date of documentation'], format = datetime_format)])
    

    train_inds = [i for (i, x) in enumerate(admission_date) if x <= np.datetime64(TRAIN_MAX_DATE)]
    val_inds = [i for (i, x) in enumerate(admission_date) if x >= (np.datetime64(TRAIN_MAX_DATE) + 1) and  x <= np.datetime64(VAL_MAX_DATE)]
    test_inds = [i for (i, x) in enumerate(admission_date) if x >= (np.datetime64(VAL_MAX_DATE)+1)]
    partition = ['train'] * len(Data)
    for i in val_inds:
        partition[i] = 'val'
    for i in test_inds:
        partition[i] = 'test'
    
    
    
    partition = pd.Series(partition)
    #################### Explanatory variables ####################
    # gender        
    gender = Data['gender'] == 'נקבה'
    gender = gender.astype('float32')


    # age
    #prev-format age = Data['Reference Event-Age At Ed Visit']
    age = Data['reference event-age when documented']
    
    age = age.astype('float32')
    
    # age_bin_num = 20
    # age_bin_labels = list(range(1,age_bin_num +1))
    # age_bin_labels = [str(round(num, 2)) for num in age_bin_labels]
    # age_bins_ranges = list(np.arange(0,1+1/(age_bin_num),1/age_bin_num)) 
    # age_bins_ranges = [round(num, 2) for num in age_bins_ranges]

    # #prev-format age = pd.qcut(Data['Reference Event-Age At Ed Visit'],q=age_bins_ranges, labels=age_bin_labels)
    
    # age = pd.qcut(Data['reference event-age when documented'],q=age_bins_ranges, labels=age_bin_labels)
    # age = age.astype('float32')


        

    # esi = Data['ESI-Emergency Severity Index (ESI)']
    esi = Data['esi_copy-emergency severity index (esi)']
    
    esi = esi.where(esi > -9999)
    esi[esi.isnull()] = 103
    esi = esi.astype('float32')
    
    # Patient complaint
    # Patient_complaint_categories_df = pd.read_excel("data/Patient Complaint Categorized.xlsx")
    # p_complaint = Data['ESI-Patient Complaint'].copy().to_frame()
    # p_complaint.columns= ['Reference Event-Patient Complaint']
    # p_complaint_categorized = p_complaint.merge(Patient_complaint_categories_df,on = 'Reference Event-Patient Complaint',how='left')
    # p_complaint_categorized = p_complaint_categorized['category']
    # p_complaint_categorized[p_complaint_categorized.isnull()] = -1
    # p_complaint_categorized = p_complaint_categorized.astype('float32')
    
    # p_complaint_categorized = pd.get_dummies(p_complaint_categorized.astype('category'),
    #                              prefix='p_complaint')

    
    
    
    # Ct-Relevant Patient Condition
    # count = CountVectorizer(binary = True)
    # ct_relevant_patient_condition = Data['Ct-Relevant Patient Condition']
    # ct_relevant_patient_condition.replace(np.nan,'EMPTY', inplace=True)
    # bow = count.fit_transform(ct_relevant_patient_condition.values)
    # ct_relevant_patient_condition = pd.DataFrame(bow.toarray(), 
                                                 # columns = ['ct_relevant_patient_condition_' + s for s in count.get_feature_names()])
    
    # Ct-Time of Order Creation I
    #prev-format ct_exam_order_creation_visit_start = pd.to_datetime(Data['Ct-Time of Order Creation'], format = datetime_format) - \
    #                  pd.to_datetime(Data['Reference Event-Visit Start Date'], format = datetime_format)
    #prev-format ct_exam_order_creation_visit_start = pd.to_datetime(Data['Ct-Time of Order Creation'], format = datetime_format) - \
                     # pd.to_datetime(Data['Reference Event-Date of Documentation'], format = datetime_format)

    
    column_name_ct_time_of_order_creation = 'ct_ed visit-time of order creation'
    if column_name_ct_time_of_order_creation not in Data.columns:
        column_name_ct_time_of_order_creation = 'ct-time of order creation'
    ct_exam_order_creation_visit_start = pd.to_datetime(Data[column_name_ct_time_of_order_creation], format = datetime_format) - \
                     pd.to_datetime(Data['reference event-date of documentation'], format = datetime_format)
    
    for i in range(n):
            ct_exam_order_creation_visit_start[i] = (ct_exam_order_creation_visit_start[i].seconds)/3600
    ct_exam_order_creation_visit_start[pd.isnull(ct_exam_order_creation_visit_start)]=-1
    ct_exam_order_creation_visit_start = ct_exam_order_creation_visit_start.astype('float32')
    # Ct-Time of Order Creation II
    
    column_name_ct_exam_start_time = 'ct_ed-exam start time'
    if column_name_ct_exam_start_time not in Data.columns:
        column_name_ct_exam_start_time = 'ct-exam start time'
    if column_name_ct_exam_start_time not in Data.columns:
        column_name_ct_exam_start_time = 'ct-exam start time'
    if column_name_ct_exam_start_time not in Data.columns:
        column_name_ct_exam_start_time = 'ct_ed visit-exam start time'
    
    ct_exam_order_creation_exam_start = pd.to_datetime(Data[column_name_ct_exam_start_time], format = datetime_format) - \
                     pd.to_datetime(Data[column_name_ct_time_of_order_creation], format = datetime_format)
    for i in range(n):
            ct_exam_order_creation_exam_start[i] = (ct_exam_order_creation_exam_start[i].seconds)/3600
    ct_exam_order_creation_exam_start[pd.isnull(ct_exam_order_creation_exam_start)]=-1
    ct_exam_order_creation_exam_start = ct_exam_order_creation_exam_start.astype('float32')
    
    #HTN-Diagnosis
    is_hyper_tension = Data['htn-diagnosis'].copy()
    for i in range(n):
        if is_hyper_tension.notnull()[i]:
            is_hyper_tension.iloc[i] = 1
        else:
            is_hyper_tension.iloc[i] = 0
    is_hyper_tension = is_hyper_tension.astype('float32')
        

    #DM-Diagnosis
    is_diabetic = Data['dm-diagnosis'].copy()
    for i in range(n):
        if is_diabetic.notnull()[i]:
            is_diabetic.iloc[i] = 1
        else:
            is_diabetic.iloc[i] = 0
    is_diabetic = is_diabetic.astype('float32')        
    
    #congestive heart failure
    is_congestive_heart_failure = Data['chf-diagnosis'].copy()
    for i in range(n):
        if is_congestive_heart_failure.notnull()[i]:
            is_congestive_heart_failure.iloc[i] = 1
        else:
            is_congestive_heart_failure.iloc[i] = 0
    is_congestive_heart_failure = is_congestive_heart_failure.astype('float32')        
    
    
    # anticoagulants-Medication
    medications_anticoagulants = Data["anticoagulants-medication"]
    # medications_anticoagulants.replace(np.nan,'EMPTY', inplace=True)
    # medications_anticoagulants = medications_anticoagulants.str.split(" ", 1, expand=True)[0]
    
    
    # medications_anticoagulants = pd.get_dummies(medications_anticoagulants.astype('category'),
    #                                             prefix='medications_anticoagulants')
    
    # hypertnesive drugs-Medication
    col_name_medications_hypertnesive = 'hypertnesive drugs-medication'
    if col_name_medications_hypertnesive not in Data.columns:
        col_name_medications_hypertnesive = 'antihypertensive-medication'  
    medications_hypertnesive = Data[col_name_medications_hypertnesive]
    # medications_hypertnesive[pd.isnull(medications_hypertnesive)] = ""
    # medications_hypertnesive = medications_hypertnesive.str.split(" ", 1, expand=True)[0]
    # medications_hypertnesive.replace(np.nan,'EMPTY', inplace=True)
    # medications_hypertnesive = pd.get_dummies(medications_hypertnesive.astype('category'),
    #                                             prefix='medications_hypertnesive')

    #smoker-Diagnosis
    
    smoking_data = Data['smoker-diagnosis'].copy()
    is_past_or_present_smoker = smoking_data.copy()
    is_present_smoker = smoking_data.copy()
    is_heavy_smoker = smoking_data.copy()
    for i in range(n):
        is_past_or_present_smoker.iloc[i] = 0
        is_present_smoker.iloc[i] = 0
        is_heavy_smoker.iloc[i] = 0
        if ('smoker' in (str)(smoking_data[i])):
            is_past_or_present_smoker.iloc[i] = 1
        
        if (('smoker' in (str)(smoking_data[i])) and ('past' not in (str)(smoking_data[i]))):
            is_present_smoker.iloc[i] = 1
        
        if (('smoker' in (str)(smoking_data[i])) and ('heavy' in (str)(smoking_data[i]))):    
            is_heavy_smoker.iloc[i] = 1
        
    is_past_or_present_smoker = is_past_or_present_smoker.astype('float32')
    is_present_smoker = is_present_smoker.astype('float32')
    is_heavy_smoker = is_heavy_smoker.astype('float32')
    Data['is past or present smoker'] = is_past_or_present_smoker
    Data['is present smoker'] = is_present_smoker
    Data['is heavy smoker'] = is_heavy_smoker
    # Data.drop(['smoker-diagnosis'], axis=1)
    
    
    #cbc wbc-Collection Date hrs
    #cbc_hgb-Collection Date hrs
    #cbc_plt-Collection Date hrs
    date_err_val = 1000
    if ('cbc wbc-collection date hrs' in Data.columns):
        cbc_date_hrs = Data['cbc wbc-collection date hrs'].copy()
        cbc_date_hrs.replace(np.nan,date_err_val, inplace=True)
        cbc_hgb_collection_date_hrs = Data['cbc_hgb-collection date hrs'].copy()
        cbc_hgb_collection_date_hrs.replace(np.nan,date_err_val,inplace=True)
        cbc_plt_collection_date_hrs = Data['cbc_plt-collection date hrs'].copy()
        cbc_plt_collection_date_hrs.replace(np.nan,date_err_val,inplace=True)
        date_hr_df = pd.DataFrame(columns=['wbc', 'hgb', 'plt'])
        date_hr_df['wbc'] = cbc_date_hrs
        date_hr_df['hgb'] = cbc_hgb_collection_date_hrs
        date_hr_df['plt'] = cbc_plt_collection_date_hrs
        min_date_hrs = date_hr_df.min(axis=1)
        Data['cbc date hrs'] = min_date_hrs.copy()
        Data.drop(['cbc wbc-collection date hrs'], axis=1)
        Data.drop(['cbc_hgb-collection date hrs'], axis=1)
        Data.drop(['cbc_plt-collection date hrs'], axis=1)
    
    if ('cbc wbc-collection date-days from reference' in Data.columns):
        cbc_date_hrs = Data['cbc wbc-collection date-days from reference'].copy()
        cbc_date_hrs.replace(np.nan,date_err_val, inplace=True)
        cbc_hgb_collection_date_hrs = Data['cbc hgb-collection date-days from reference'].copy()
        cbc_hgb_collection_date_hrs.replace(np.nan,date_err_val,inplace=True)
        
        col_name_cbc_plt_collection_date_hrs = 'cbc_plt-collection date-days from reference'
        if col_name_cbc_plt_collection_date_hrs not in Data.columns:
            col_name_cbc_plt_collection_date_hrs = 'cbc plt-collection date-days from reference'
        cbc_plt_collection_date_hrs = Data[col_name_cbc_plt_collection_date_hrs].copy()
        cbc_plt_collection_date_hrs.replace(np.nan,date_err_val,inplace=True)
        date_hr_df = pd.DataFrame(columns=['wbc', 'hgb', 'plt'])
        date_hr_df['wbc'] = cbc_date_hrs
        date_hr_df['hgb'] = cbc_hgb_collection_date_hrs
        date_hr_df['plt'] = cbc_plt_collection_date_hrs
        min_date_hrs = date_hr_df.min(axis=1)
        Data['cbc date hrs'] = min_date_hrs.copy()
    cbc_from_visit_start_med = min_date_hrs.median()
    cbc_from_visit_start = min_date_hrs.copy()
    cbc_from_visit_start.loc[cbc_from_visit_start == date_err_val] = cbc_from_visit_start_med #Median


    quantile_wbc = Data['cbc wbc-numeric result'].copy()
 
    #cbc_hgb-Numeric Result
    col_name_cbc_hgb_numeric = 'cbc hgb-numeric result'
    if col_name_cbc_hgb_numeric not in Data.columns:
        col_name_cbc_hgb_numeric = 'cbc_hgb-numeric result'
    quantile_hgb  = Data[col_name_cbc_hgb_numeric].copy()
    
    
    
    #cbc_plt-Numeric Result
    col_name_cbc_plt_numeric_result = 'cbc_plt-numeric result'
    if col_name_cbc_plt_numeric_result not in Data.columns:
        col_name_cbc_plt_numeric_result = 'cbc plt-numeric result'
    quantile_plt = Data[col_name_cbc_plt_numeric_result].copy()

    
    
    #temperature
    # temperature_numeric_result = Data['temp-numeric result']
    quantile_temperature = Data['temp-numeric result'].copy()

    
    
    #pulse
    # pulse_numeric_result = Data['pulse-numeric result']
    quantile_pulse = Data['pulse-numeric result'].copy()

    
    #BP SYS
    quantile_BP_SYS = Data['bp sys-numeric result'].copy()

    
    #BP DIAS
    quantile_BP_DIAS = Data['bp dias-numeric result'].copy()

    
    #CRP
    quantile_CRP = Data['crp-numeric result'].copy()

    
    #troponin
    quantile_troponin = Data['troponin-numeric result'].copy()
    
    
    # Ct-Exam Performed (SPS)
    column_name_ct_exam_performed = 'ct_ed visit-exam performed (sps)'
    if column_name_ct_exam_performed not in Data.columns:
        column_name_ct_exam_performed = 'ct-exam performed (sps)'
        
    column_name_ct_done = 'ct_ed visit-interpretation'
    if column_name_ct_done not in Data.columns:
        column_name_ct_done = 'ct-interpretation'
    ct_exam_performed = Data[column_name_ct_exam_performed]
    ct_done = [] 
    for i in range(len(Data)):
        ct_done.append((type(Data[column_name_ct_done][i]) == str) * 1.)
    ct_done = pd.Series(ct_done)
    

    # Ct-Exam Performed (SPS)
    exam_performed_type = pd.Series(['EMPTY'] * len(Data))
    exam_performed_subtype = pd.Series(['EMPTY'] * len(Data))
    exam_performed_res = pd.Series([''] * len(Data))
    all_exam_performed_res = pd.Series([''] * len(Data))
    
    col_name_ct_interpretation = None
    col_name_ct_interpretation_variations_by_priority = [
        'ct_ed visit-interpretation', 'ct-interpretation']
    for d_c in col_name_ct_interpretation_variations_by_priority:
        if d_c in Data.columns:
            col_name_ct_interpretation = d_c
            break;
            
    for d_c in col_name_ct_interpretation_variations_by_priority:
        if d_c in Data.columns:
            news = Data[d_c].copy()
            news = news.fillna("")
            for i in range(len(news)):
                news[i] = get_text_summary(news[i], DEFAULT_TEXT_MAX_SENTENCES)
            all_exam_performed_res = all_exam_performed_res.str.cat(news, sep=".") 
            all_exam_performed_res = all_exam_performed_res.fillna("")
            
    col_name_us_interpretation =  None
    col_name_us_interpretation_variations_by_priority = [
        'usg ed-interpretation']
    for d_c in col_name_us_interpretation_variations_by_priority:
        if d_c in Data.columns:
            col_name_us_interpretation = d_c
            break;
            
    for d_c in col_name_us_interpretation_variations_by_priority:
        if d_c in Data.columns:
            news = Data[d_c].copy()
            news = news.fillna("")
            for i in range(len(news)):
                news[i] = get_text_summary(news[i], DEFAULT_TEXT_MAX_SENTENCES)
            all_exam_performed_res = all_exam_performed_res.str.cat(news, sep=".")
            all_exam_performed_res = all_exam_performed_res.fillna("")
    
    col_name_xray_interpretation =  None
    col_name_xray_interpretation_variations_by_priority = [
   'אגן x ray-interpretation',
        'מפרק ירך x ray-interpretation',
        'chest x ray-interpretation', 'chest x ray_copy-interpretation',
        'x ray-interpretation']
    for d_c in col_name_xray_interpretation_variations_by_priority:
        if d_c in Data.columns:
            col_name_xray_interpretation = d_c
            break;
    for d_c in col_name_xray_interpretation_variations_by_priority:
        if d_c in Data.columns:
            news = Data[d_c].copy()
            news = news.fillna("")
            for i in range(len(news)):
                news[i] = get_text_summary(news[i], DEFAULT_TEXT_MAX_SENTENCES)
            all_exam_performed_res = all_exam_performed_res.str.cat(news, sep=".")
            all_exam_performed_res = all_exam_performed_res.fillna("")
            
    ct_exam_res = pd.Series([''] * len(Data))
    us_exam_res = pd.Series([''] * len(Data))
    xray_exam_res = pd.Series([''] * len(Data))
    
    if col_name_ct_interpretation is not None:
        ct_exam_res = Data[col_name_ct_interpretation].copy()
    else:
        print("DID NOT FIND CT INTERPRETATION COL IN FILE: " + filename)
    if col_name_us_interpretation is not None:
        us_exam_res = Data[col_name_us_interpretation].copy()
    else:
        print("DID NOT FIND US INTERPRETATION COL IN FILE: " + filename)
    if col_name_xray_interpretation is not None:
        xray_exam_res = Data[col_name_xray_interpretation].copy()
    else:
        print("DID NOT FIND X-RAY INTERPRETATION COL IN FILE: " + filename)
    
    for i in range(len(Data)):
        if ('usg ed-interpretation' in Data.columns):
            if not pd.isna(Data['usg ed-interpretation'].iloc[i]):
                exam_performed_type[i] = 'US'
                exam_performed_res[i] = Data['usg ed-interpretation'].iloc[i]
        if ('usg ed-exam performed (sps)' in Data.columns):
            if not pd.isna(Data['usg ed-exam performed (sps)'].iloc[i]):
                exam_performed_subtype[i] = Data['usg ed-exam performed (sps)'].iloc[i]
        
        if ('x ray-interpretation' in Data.columns):
            if not pd.isna(Data['x ray-interpretation'].iloc[i]):
                exam_performed_type[i] = 'X-RAY'
                exam_performed_res[i] = Data['x ray-interpretation'].iloc[i]
        if ('x ray-exam performed (sps)' in Data.columns):
            if not pd.isna(Data['x ray-exam performed (sps)'].iloc[i]):
                exam_performed_subtype[i] = Data['x ray-exam performed (sps)'].iloc[i]

        if ('ct_ed visit-interpretation' in Data.columns):
            if not pd.isna(Data['ct_ed visit-interpretation'].iloc[i]):
                exam_performed_type[i] = 'CT'
                exam_performed_res[i] = Data['ct_ed visit-interpretation'].iloc[i]
        if ('ct_ed visit-exam performed (sps)' in Data.columns):
            if not pd.isna(Data['ct_ed visit-exam performed (sps)'].iloc[i]):
                exam_performed_subtype[i] = Data['ct_ed visit-exam performed (sps)'].iloc[i]
        


    
    
    
    return dtd_le_30, dtd_le_60, dtd_le_90, dtd_le_120,\
            readmission_in_30_days, readmission_in_60_days, readmission_in_90_days, readmission_in_120_days,\
            days_to_first_readmission,days_to_death,admission_date,partition,readmission_class,\
            gender,age,esi, ct_exam_order_creation_visit_start,is_past_or_present_smoker,\
            is_present_smoker,quantile_wbc, quantile_hgb,quantile_plt,\
            cbc_from_visit_start,cbc_plt_collection_date_hrs,is_hyper_tension,is_diabetic,\
            is_congestive_heart_failure,\
            ct_exam_order_creation_exam_start,ct_done, \
            ct_exam_performed, medications_anticoagulants, medications_hypertnesive, diagnosis, \
            exam_performed_type, exam_performed_subtype, exam_performed_res, \
            quantile_temperature, quantile_pulse, quantile_BP_SYS,quantile_BP_DIAS,\
            quantile_CRP, quantile_troponin,\
            ct_exam_res, us_exam_res, xray_exam_res, all_exam_performed_res


def get_quantiles(numeric_res, bin_num, lower_boundary, upper_boundery):
    bin_labels = list(range(1,bin_num +1))
    bin_labels = [str(round(num, 2)) for num in bin_labels]
    bins_ranges = list(np.arange(0,1+1/(bin_num),1/bin_num)) 
    bins_ranges = [round(num, 2) for num in bins_ranges]
    
    
    # quantiles = numeric_res.copy()

    numeric_res_med = numeric_res.median()
    print("numeric_res_med"+ str(numeric_res_med))
    numeric_res.fillna(lower_boundary-1.0, inplace= True)
    numeric_res.mask(numeric_res == np.nan,lower_boundary-1, inplace=True)
    numeric_res.mask(numeric_res < lower_boundary, numeric_res_med, inplace=True)
    numeric_res.mask(numeric_res > upper_boundery, numeric_res_med, inplace=True)
    numeric_result_ranked = numeric_res.rank(method = 'first')
    quantiles = pd.qcut(numeric_result_ranked,q=bins_ranges, labels=bin_labels, duplicates = 'drop')
    # quantile_wbc = Data['quantile_wbc'].copy()
    quantiles = quantiles.astype('float32')
    return quantiles


def get_grouped_diagnosis(ill_type, diagnosis):
    grouped_diagnosis = diagnosis.copy()
    grouped_diagnosis = grouped_diagnosis.replace(regex=[r'.*appendicit.*'], value='appendicit')
    
    grouped_diagnosis = grouped_diagnosis.replace(regex=[r'.*cva.*'], value='stroke')
    grouped_diagnosis = grouped_diagnosis.replace(regex=[r'.*stroke.*'], value='stroke')
    grouped_diagnosis = grouped_diagnosis.replace(regex=[r'.*ischemic.*'], value='stroke')
    inds = grouped_diagnosis == 'stroke'
    grouped_diagnosis[inds] = (ill_type[inds]).str.cat((grouped_diagnosis[inds]).values.astype(str), sep=' ')
    grouped_diagnosis = grouped_diagnosis.replace(regex=[r'.*ICHfinilizing.*'], value='ICH stroke') 
    grouped_diagnosis = grouped_diagnosis.replace(regex=[r'.*Ischemicstroke.*'], value='ischemic stroke') 
    
    grouped_diagnosis = grouped_diagnosis.replace(regex=[r'.*femur.*'], value='femoral fracture')
    return grouped_diagnosis

dtd_le_30, dtd_le_60, dtd_le_90, dtd_le_120,\
readmission_in_30_days, readmission_in_60_days, readmission_in_90_days, readmission_in_120_days,\
days_to_first_readmission,days_to_death,admission_date,partition,readmission_class,\
gender,age,esi, ct_exam_order_creation_visit_start,is_past_or_present_smoker,\
is_present_smoker,quantile_wbc, quantile_hgb,quantile_plt,\
cbc_from_visit_start,cbc_plt_collection_date_hrs,is_hyper_tension,is_diabetic,\
is_congestive_heart_failure,\
ct_exam_order_creation_exam_start,ct_done, \
ct_exam_performed, medications_anticoagulants, medications_hypertnesive, diagnosis, ill_type, \
exam_performed_type, exam_performed_subtype, exam_performed_res, \
quantile_temperature, quantile_pulse, quantile_BP_SYS,quantile_BP_DIAS,\
quantile_CRP, quantile_troponin, \
ct_exam_res, us_exam_res, xray_exam_res, all_exam_performed_res = get_all_files_features_and_outcomes(filenames)


dtd_le_30 = dtd_le_30.reset_index()[0]
dtd_le_60 = dtd_le_60.reset_index()[0]
dtd_le_90 = dtd_le_90.reset_index()[0]
dtd_le_120 = dtd_le_120.reset_index()[0]
readmission_in_30_days = readmission_in_30_days.reset_index()[0]
readmission_in_60_days = readmission_in_60_days.reset_index()[0]
readmission_in_90_days = readmission_in_90_days.reset_index()[0]
readmission_in_120_days = readmission_in_120_days.reset_index()[0]
days_to_first_readmission = days_to_first_readmission.reset_index()[0]
days_to_death = days_to_death.reset_index()[0]
admission_date= admission_date.reset_index()[0]
partition = partition.reset_index()[0]
readmission_class = readmission_class.reset_index()[0]
gender = gender.reset_index()[0]

esi = esi.reset_index()[0]
ct_exam_order_creation_visit_start = ct_exam_order_creation_visit_start.reset_index()[0]
is_past_or_present_smoker = is_past_or_present_smoker.reset_index()[0]
is_present_smoker = is_present_smoker.reset_index()[0]

cbc_from_visit_start = cbc_from_visit_start.reset_index()[0]
cbc_plt_collection_date_hrs = cbc_plt_collection_date_hrs.reset_index()[0]
is_hyper_tension = is_hyper_tension.reset_index()[0]
is_diabetic = is_diabetic.reset_index()[0]
is_congestive_heart_failure = is_congestive_heart_failure.reset_index()[0]
ct_exam_order_creation_exam_start = ct_exam_order_creation_exam_start.reset_index()[0]
ct_done = ct_done.reset_index()[0]
ct_exam_performed = ct_exam_performed.reset_index()[0]
medications_anticoagulants = medications_anticoagulants.reset_index()[0]
medications_hypertnesive = medications_hypertnesive.reset_index()[0]
diagnosis = diagnosis.reset_index()[0]
ill_type = ill_type.reset_index()[0]
exam_performed_type = exam_performed_type.reset_index()[0]
exam_performed_subtype = exam_performed_subtype.reset_index()[0]
exam_performed_res = exam_performed_res.reset_index()[0]


num_bins = 10

quantile_age = age.reset_index()[0]
quantile_age = get_quantiles(quantile_age, num_bins, 0, 200)


quantile_wbc = quantile_wbc.reset_index()[0]
quantile_wbc = get_quantiles(quantile_wbc, num_bins, 2, 100)

quantile_hgb = quantile_hgb.reset_index()[0]
quantile_hgb = get_quantiles(quantile_hgb, num_bins, 3, 20)

quantile_plt = quantile_plt.reset_index()[0]
quantile_plt = get_quantiles(quantile_plt, num_bins, 10, 700)

quantile_temperature = quantile_temperature.reset_index()[0]
quantile_temperature = get_quantiles(quantile_temperature, num_bins, 34, 44)

quantile_pulse = quantile_pulse.reset_index()[0]
quantile_pulse = get_quantiles(quantile_pulse, num_bins, 30, 200)

quantile_BP_SYS = quantile_BP_SYS.reset_index()[0]
quantile_BP_SYS = get_quantiles(quantile_BP_SYS, num_bins, 40, 240)

quantile_BP_DIAS = quantile_BP_DIAS.reset_index()[0]
quantile_BP_DIAS = get_quantiles(quantile_BP_DIAS, num_bins, 20, 250)

quantile_CRP = quantile_CRP.reset_index()[0]
quantile_CRP = get_quantiles(quantile_CRP, num_bins, 0, 200)

quantile_troponin = quantile_troponin.reset_index()[0]
quantile_troponin = get_quantiles(quantile_troponin, num_bins, 0, 10000)


ct_exam_performed.replace(np.nan,'EMPTY', inplace=True)
ct_exam_performed = pd.get_dummies(ct_exam_performed.astype('category'), prefix='ct_exam_performed', prefix_sep=' ')


diagnosis_org = diagnosis.copy()
diagnosis_org_distinct_vals = diagnosis_org.value_counts()
diagnosis_total_count = diagnosis_org_distinct_vals.sum()
for idiag in range(len(diagnosis_org_distinct_vals)):
    if (diagnosis_org_distinct_vals[idiag]/diagnosis_total_count < (0.05/len(filenames))):
        diagnosis_org.loc[diagnosis_org == diagnosis_org_distinct_vals.index[idiag]] = "Too low frequency diagnosis"
diagnosis = pd.get_dummies(diagnosis_org.astype('category'), prefix='Diagnosis:', prefix_sep=' ') 
grouped_diagnosis_org = get_grouped_diagnosis(ill_type, diagnosis_org)
grouped_diagnosis = pd.get_dummies(grouped_diagnosis_org.astype('category'), prefix='Diagnosis:', prefix_sep=' ') 



medications_anticoagulants.replace(np.nan,'None', inplace=True)
medications_anticoagulants = medications_anticoagulants.str.split(" ", 1, expand=True)[0]
medications_anticoagulants = pd.get_dummies(medications_anticoagulants.astype('category'), prefix='Medications anticoagulants:', prefix_sep=' ')

medications_hypertnesive[pd.isnull(medications_hypertnesive)] = ""
medications_hypertnesive = medications_hypertnesive.str.split(" ", 1, expand=True)[0]
medications_hypertnesive.replace(np.nan,'None', inplace=True)
medications_hypertnesive = pd.get_dummies(medications_hypertnesive.astype('category'),
                                            prefix='Medications hypertnesive:', prefix_sep=' ')

ill_type = pd.get_dummies(ill_type.astype('category'), prefix='Ill type:', prefix_sep=' ')

exam_performed_type = pd.get_dummies(exam_performed_type.astype('category'), prefix='Exam performed type:', prefix_sep=' ')
#################### Organize and save data frames ####################
    
outcomes = {'dtd_le_30'                 : dtd_le_30,
            'dtd_le_60'                 : dtd_le_60,
            'dtd_le_90'                 : dtd_le_90,
            'dtd_le_120'                : dtd_le_120,
            'readmissions_in_30_days'   : readmission_in_30_days,
            'readmissions_in_60_days'   : readmission_in_60_days,
            'readmissions_in_90_days'   : readmission_in_90_days,
            'readmissions_in_120_days'  : readmission_in_120_days,
            'days_to_first_readmission' : days_to_first_readmission,
            'days_to_death'             : days_to_death,
            'admission_date'            : admission_date,
            'partition'                 : partition
           }


if ((Counter(dtd_le_30)[0] < 30) |(Counter(dtd_le_30)[1] < 30)): #highly imbalanced data
    print("Warning: Highly imbalanced data dtd_le_30 replacing with dtd_le_360")

    outcomes = {'dtd_le_120_W'                 : dtd_le_120,
        'dtd_le_60'                 : dtd_le_60,
        'dtd_le_90'                 : dtd_le_90,
        'dtd_le_120'                : dtd_le_120,
        'readmissions_in_30_days'   : readmission_in_30_days,
        'readmissions_in_60_days'   : readmission_in_60_days,
        'readmissions_in_90_days'   : readmission_in_90_days,
        'readmissions_in_120_days'  : readmission_in_120_days,
        'days_to_first_readmission' : days_to_first_readmission,
        'days_to_death'             : days_to_death,
        'admission_date'            : admission_date,
        'partition'                 : partition
       }

if ((Counter(readmission_in_30_days)[0] < 30) | (Counter(readmission_in_30_days)[1] < 30)): #highly imbalanced data
    print("Warning: Highly imbalanced data readmissions_in_30_days replacing with readmissions_in_120_days")
    outcomes = {'dtd_le_30'                 : dtd_le_30,
            'dtd_le_60'                 : dtd_le_60,
            'dtd_le_90'                 : dtd_le_90,
            'dtd_le_120'                : dtd_le_120,
            'readmissions_in_120_days_W'   : readmission_in_120_days,
            'readmissions_in_60_days'   : readmission_in_60_days,
            'readmissions_in_90_days'   : readmission_in_90_days,
            'readmissions_in_120_days'  : readmission_in_120_days,
            'days_to_first_readmission' : days_to_first_readmission,
            'days_to_death'             : days_to_death,
            'admission_date'            : admission_date,
            'partition'                 : partition
           }

outcomes2 = {
    # 'dtd_class'               : dtd_class,
             'readmission_class'       : readmission_class,
             'readmissions_in_120_days'   : readmission_in_120_days,

           }

outcomes = pd.DataFrame(outcomes)
outcomes2 = pd.DataFrame(outcomes2)

assert outcomes.isna().sum().all() == 0
assert outcomes2.isna().sum().all() == 0
assert outcomes2.isnull().values.any() == False  
outcomes.to_pickle('data/outcomes.pkl')
outcomes2.to_pickle('data/outcomes2.pkl')


X = {'gender'                                : gender,
    # 'age'                                    : age,
    'quantile age'                          : quantile_age,
    'esi'                                    : esi,
    # 'ct_exam_order_creation_visit_start'     : ct_exam_order_creation_visit_start,
    'is past or present smoker'              : is_past_or_present_smoker,
    'is present smoker'                      : is_present_smoker,
    'quantile wbc'                          : quantile_wbc,
    'quantile hgb'                          : quantile_hgb,
    'quantile plt'                          : quantile_plt,
    #'cbc from visit start'                   : cbc_from_visit_start,
    # 'cbc_plt_collection_date_hrs'            : cbc_plt_collection_date_hrs,
    'is hyper tension'                       : is_hyper_tension,
    'is diabetic'                            : is_diabetic,
    'is congestive heart failure'            : is_congestive_heart_failure,
#    'delta_time_ct_exam_from_visit_start'    : ct_exam_order_creation_visit_start,
    # 'delta_time_ct_exam_start_from_creation' : ct_exam_order_creation_exam_start,
    # 'ct_done'                                : ct_done,
    'quantile temperature'                   : quantile_temperature,
    'quantile pulse'                         : quantile_pulse,
    'quantile BP SYS'                        : quantile_BP_SYS,
    'quantile BP DIAS'                       : quantile_BP_DIAS,
    'quantile CRP'                           : quantile_CRP,
    'quantile troponin'                      : quantile_troponin

    }




X = pd.DataFrame(X)

X = pd.concat([X, 
               # ct_exam_performed,
               medications_anticoagulants,
               medications_hypertnesive,
               #diagnosis,
               # exam_performed_type,
               #ill_type,
              ]
              , axis=1)

X_Categories = pd.concat([ill_type,
                            # diagnosis,
                            # grouped_diagnosis,
                          ], axis = 1)

assert X.isna().values.any() == False    
assert X.isnull().values.any() == False  
X.to_pickle('data/preprocessed_X.pkl')
assert X_Categories.isna().values.any() == False
assert X_Categories.isnull().values.any() == False 
X_Categories.to_pickle('data/X_Categories.pkl') 


text_segmented = get_text_segments((exam_performed_res))
text_segmented.to_csv('data/segmented_text.csv', sep='\t')
ct_exam_res_segmented = get_text_segments(ct_exam_res)
ct_exam_res_segmented.to_csv('data/segmented_ct_text.csv', sep='\t')
us_exam_res_segmented = get_text_segments(us_exam_res)
us_exam_res_segmented.to_csv('data/segmented_us_text.csv', sep='\t')
xray_exam_res_segmented = get_text_segments(xray_exam_res)
xray_exam_res_segmented.to_csv('data/segmented_xray_text.csv', sep='\t')






Text = {
        'Free-text-ct-exam-res' : ct_exam_res_segmented['original_text'],
        'Free-text-us-exam-res' : us_exam_res_segmented['original_text'],
        'Free-text-x-ray-exam-res' : xray_exam_res_segmented['original_text'],
        'Free-text-all-exam-res' : all_exam_performed_res
        # 'Free-text-all-exam-res' : ct_exam_res_segmented['original_text'] + " " + us_exam_res_segmented['original_text'] + " " + xray_exam_res_segmented[]
            # exam_performed_res
        # 'diagnosis': diagnosis_org,
        }
Text = pd.DataFrame(Text)
Text.to_pickle('data/text.pkl')








