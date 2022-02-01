# -*- coding: utf-8 -*-
"""
Created on Tue Feb  1 11:28:02 2022

@author: srandrad
"""

import pandas as pd
import os

test_df1 = pd.DataFrame({"Test_col":[1,2,3,4],
                         "Test_col_2":[1,2,3,4]})

file = os.path.join('results','test.xlsx')
test_df1.to_excel(file, sheet_name= "test1")

test_df2 = pd.DataFrame({"Test_col":[4,5,6,7],
                         "Test_col_2":[1,2,3,4]})

with pd.ExcelWriter(file, engine='openpyxl', mode='a') as writer:  
    test_df2.to_excel(writer, sheet_name="test2")

test_df1.to_csv(os.path.join('results','test1.csv'))