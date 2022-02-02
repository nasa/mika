# -*- coding: utf-8 -*-
"""
Created on Mon Jan 31 11:17:44 2022

SAFECOM Web Scraper

@author: srandrad
"""

from bs4 import BeautifulSoup
import pandas as pd
import os
from selenium import webdriver
import time

safecom_page = "https://www.safecom.gov/safecom/"
#ids are of form "XX-XXXX" ex: 94-0001
years = ["94", "95", "96", "97", "98","99", "00", "01", "02", "03", 
         "04", "05", "06", "07", "08", "09", "10", "11", "12", "13", "14", "15", "16", "17",
         "18", "19", "20", "21", "22"]
max_reports_per_year = {"94":"0005", "95":"0287", "96":"0096", "97":"262", "98":"0847", 
                        "99":"1062", "00":"1402", "01":"1154", "02":"1482", "03":"1433", 
                        "04": "0913", "05":"0929", "06":"1297", "07":"1118", "08":"0972", 
                        "09":"0839", "10":"0780", "11": "1048", "12":"1129", "13":"0960", 
                        "14":"0825", "15":"0868", "16":"1080", "17":"0984", "18":"1003", 
                        "19":"0642", "20":"1317", "21":"1040", "22":"0093"}

possible_report_nums = []
for d1 in range(0,10):
    for d2 in range(0,10):
        for d3 in range(0,10):
            for d4 in range(1,10):
                possible_report_nums.append(str(d1)+str(d2)+str(d3)+str(d4))

event_data = {"Agency":[], "Unit":[], "Region":[], "Location":[], "Date":[], 
              "Local Time":[], "Date Submitted":[], "Tracking #":[]}
mission_data = {"Mission Type":[], "Procurement Type":[], "Persons Onboard":[],
                "Departure Point":[], "Destination":[], "Special Use":[],
                "Damages":[], "Injuries":[], "Hazardous Materials":[],
                "Other Procurement Type":[], "Other Mission Type":[]}
aircraft_data = {"Type":[], "Manufacturer":[], "Model":[]}
text_data = {"Narrative":[], "Corrective Action":[]}
categories_data = {"Hazard":[], "Incident":[], "Management":[], "UAS":[],
                   "Accident":[], "Airspace":[], "Maintenance":[], "Mishap Prevention":[]}
total_data = {**event_data, **mission_data, **aircraft_data, **text_data, **categories_data}

def collect_data(url, total_data, t=1, retry=False):
    browser = webdriver.Chrome()
    browser.get(url)
    time.sleep(t)
    results = browser.page_source
    browser.quit()

    soup = BeautifulSoup(results, "html.parser")
    #check page is valid:
    error = soup.findAll("div", class_='error-container')
    if len(error)>0:
        return total_data
    current_cats = {cat:[] for cat in categories_data}
    for group in soup.findAll("div", class_="item-value"):
        if (len(group.findAll("div", class_="item-value"))) > 0:
            continue
        label = group.select('label')[0].text.strip(": ")
        value = group.select('span')[0].text
        if value is None: value = ""
        if label in categories_data: 
            current_cats[label].append(value)
        else:
            total_data[label].append(value)
        
    #getting narrative
    narrative = soup.find("div", class_='narrative-row')
    if narrative is None:
        if retry == True: 
            print("error on ", url)
            return
        total_data = collect_data(url, total_data, t=3, retry=True)
        return total_data
    total_data['Narrative'].append(narrative.text)
    #getting corrective action
    corrective_action = soup.find("div", class_='corrective-row')
    total_data['Corrective Action'].append(corrective_action.text)
    
    #handeling missing values and multiple values in categories data
    for cat in categories_data:
        if current_cats[cat] == []:
            total_data[cat].append("")
        else:
            total_data[cat].append(", ".join(current_cats[cat]))
            
    return total_data

for year in years:
    #if year == "19":
    #    start_num = "0025"
    #    report_nums = [num for num in possible_report_nums if int(num)>=int(start_num)]
    #else:
    report_nums = possible_report_nums
    for num in report_nums:
        if int(num) > int(max_reports_per_year[year]):
            break
        else:
            report_id = str(year)+"-"+str(num)
            url = safecom_page+report_id
            total_data = collect_data(url, total_data)
        
safecom_df = pd.DataFrame(total_data)
safecom_data_so_far = pd.read_csv(os.path.join('data','SAFECOM_data_v6.csv'), index_col=0)
safecom_df = pd.concat([safecom_data_so_far, safecom_df]).reset_index(drop=True)
safecom_df.to_csv(os.path.join('data','SAFECOM_data.csv'))