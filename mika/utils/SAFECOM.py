# -*- coding: utf-8 -*-
"""
Created on Thu Aug 11 13:14:21 2022

@author: srandrad
"""

#safecom utils
def get_severity_FAA(severities): #SAFECOM
    curr_severities = {hazard:0 for hazard in severities}
    for hazard in severities:
        s = severities[hazard]
        if s<=0.1: #negligible impact
            severity = 'Minimal Impact'
        elif s>0.1 and s <= 0.5:
            severity = 'Minor Impact'
        elif s>0.5 and s<=1:
            severity = 'Major Impact'
        elif s>1 and s<=2:
            severity = 'Hazardous Impact'
        elif s>2:
            severity = 'Catastrophic Impact'
        curr_severities[hazard] = severity
    return curr_severities


def get_severity_USFS(severities):
    curr_severities = {hazard:0 for hazard in severities}
    for hazard in severities:
        s = severities[hazard]
        if s<=0.1: #negligible impact
            severity = 'Negligible'
        elif s>0.1 and s <= 1:
            severity = 'Marginal'
        elif s>1 and s<=2:
            severity = 'Critical'
        elif s>2:
            severity = 'Catastrophic'
        curr_severities[hazard] = severity
    return curr_severities