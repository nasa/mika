# -*- coding: utf-8 -*-
"""
Created on Thu Mar  4 10:27:55 2021

@author: srandrad
"""


from requests import get
from bs4 import BeautifulSoup
import pandas as pd
import numpy as np
import re
import time

terms = []

#pages = ["List_of_national_forests_of_the_United_States"]
#pages = ["List_of_national_parks_of_the_United_States"]
#pages = ["National_grassland"]
#page = "List_of_rivers_of_the_United_States"
#letters = ["_A", "_B", "_C","_D", "_E", "_F", "_G", "_H", "_I", "_J", "_K", "_L", "_M",
#           "_N", "_O", "_P", "_Q", "_R", "_S", "_T", "_U", "_V", "_W", "_XYZ"]

#page = "https://en.wikipedia.org/wiki/List_of_Alabama_state_forests"
page = "https://en.wikipedia.org/wiki/Lists_of_state_parks_by_U.S._state"

states = ["Alabama", "Alaska", #"Arizona",
          #"Arkansas",
          "California", #"Colorado" uses ul, 
          "Connecticut", "Delaware", "Florida", "Georgia", #"Hawaii", 
          "Idaho", "Illinois",
          "Indiana", "Iowa", #"Kansas",
          "Kentucky", "Louisiana", "Maine", "Maryland",
          "Massachusetts", #"Michigan", uses ul
          "Minnesota", #"Mississippi",
          "Missouri", 
          "Montana", #"Nebraska", "Nevada", 
          "New Hampshire","New Jersey", #"New Mexico", 
          "New York", "North Carolina", #"North Dakota", 
          "Ohio", #"Oklahoma", 
          "Oregon", "Pennsylvania", "Rhode Island", "South Carolina",# "South Dakota",
          "Tennessee",
          "Texas", #"Utah", 
          "Vermont", "Virginia", "Washington", "West Virginia", "Wisconsin"]#, "Wyoming"]

page = "https://en.wikipedia.org/wiki/List_of_United_States_counties_and_county_equivalents"
results = get(page)
soup = BeautifulSoup(results.text, "html.parser")
div = soup.find("div", class_="mw-parser-output")
tbody = div.find("tbody")
trs = tbody.find_all("tr")
for tr in trs:
    tds = tr.find_all("td")
    print(tds)
    if tds != []:
        td = tds[0]
        county = td.a.text.split(",")
        county = county[0].replace("Municipality", "")
        terms.append(county)

print(terms)

"""
for state in states:
    print(state)
    url = "https://en.wikipedia.org/wiki/List_of_"+state+"_state_forests"
    results = get(url)
    soup = BeautifulSoup(results.text, "html.parser")
    #print(soup)
    div = soup.find("div", class_="mw-parser-output")
    time.sleep(3)
    #print(div)
    #print(div)
    #table = soup.find("table", class_="wikitable sortable plainrowheaders jquery-tablesorter")
    
    #print(div)
    #print (table)
    #print(div)
    body = div.find_all("tbody")
    #print(len(body), body[0])
    #body=body[1]
    n=0
    #rows = []
    bod_num = 0
    if state != "Massachusetts" and state != "Michigin": body = [body[0]]
    if state == "Massachusetts": body = [body[1]]
    for bod in body:
        if (state == "California" or state == "Connecticut") and bod_num>0: 
            break
        trs = bod.find_all("tr")
        n = 0
        for tr in trs:
            if n>0:
                #print(tr)
                tds = tr.find_all("td")
                #if state == "Connecticut": print(tds)
                if tds != []:
                    td = tds[0]
                    #print(td.text)
                    #print(len(td))
                    terms.append(td.text.replace(" State", "").replace(" Forest","").strip("\n"))
            n+=1
        bod_num+=1
    #    list_of_rows = bod.find_all(scope="row")
    #    for row in list_of_rows:
    #        rows.append(row)
    #rows = [row for row in body.find_all(scope="row") for body in body]
    #print(rows)
    #uls = div[0].find_all("ul")
    #uls = div.find_all("ul")
    #print(uls, type(uls))
    #for ul in uls:
        #print(tr)
        #print(ul)
        #if ul != uls[0]:
            #lis = ul.find_all("li")
            #print(lis)
            #for li in lis:
                #print(li.a.text)
                #if li.text != "Harbor River - South Carolina":
                 #   raw_text = li.a.text.replace(" River", "").replace(" Creek", "")
                    #clean_text = strin = re.sub(" [\(\[].*?[\)\]]", "", raw_text)
                    #terms.append(clean_text)
                    #n+=1
                #else: terms.append("Harbor")
                #print(n)
            
terms = [t for term in terms for t in term.split("\n")]
print(terms,len(terms))
"""

"""
terms_cleaned = []
for t in terms:
    cleaned = (t.lower().replace('"',""))
    #print(cleaned)
    cleaned_t = re.sub(r'[(]((\w)*(\s)*)*[)]', "", cleaned)
    #print(cleaned_t)
    terms_cleaned.append(cleaned_t)
terms = list(set(terms_cleaned))
print(terms)
engineering = []
for t in terms:
    engineering.append(1)

lexicon = pd.DataFrame({
    "rating":engineering,
    "term": terms
    })
lexicon.to_csv(path_or_buf = "_general_engineering_lexicon.csv", 
               columns = ["rating", "term"], index = False)
"""
