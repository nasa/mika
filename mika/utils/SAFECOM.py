# -*- coding: utf-8 -*-
"""
Created on Thu Aug 11 13:14:21 2022

@author: srandrad
"""

#safecom utils
def get_SAFECOM_severity_FAA(severities): #SAFECOM
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


def get_SAFECOM_severity_USFS(severities):
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

def get_UAS_likelihood_FAA(frequency):
    curr_likelihoods = {hazard:0 for hazard in frequency}
    for hazard in frequency:
        r = frequency[hazard]
        if r==5:
            likelihood = 'Frequent'
        elif r==4:
            likelihood = 'Probable'
        elif r==3:
            likelihood = 'Remote'
        elif r==2:
            likelihood = 'Extremely Remote'
        elif r==1:
            likelihood = 'Extremely Improbable'
        curr_likelihoods[hazard] = likelihood
    return curr_likelihoods

def get_UAS_likelihood_USFS(frequency):
    curr_likelihoods = {hazard:0 for hazard in frequency}
    for hazard in frequency:
        r = frequency[hazard]
        if r==5:
            likelihood = 'Frequent'
        elif r==4:
            likelihood = 'Probable'
        elif r==3:
            likelihood = 'Occasional'
        elif r==2:
            likelihood = 'Remote'
        elif r==1:
            likelihood = 'Improbable'
        curr_likelihoods[hazard] = likelihood
    return curr_likelihoods