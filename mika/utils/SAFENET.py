# -*- coding: utf-8 -*-
"""
Created on Thu Aug 11 13:14:08 2022

@author: srandrad
"""

#safenet utils
def get_SAFENET_severity_FAA(severities):
    """
    Assigns a severity category according to FAA risk matrix

    Parameters
    ----------
    severities : dict
        Dictionary with keys as hazards and values as average severities.

    Returns
    -------
    curr_severities : dict
        Dictionary with keys as hazards and values as severity category.

    """
    curr_severities = {hazard:0 for hazard in severities}
    for hazard in severities:
        s = severities[hazard]
        if s== 1: #negligible impact
            severity = 'Minimal Impact'
        elif s==2:
            severity = 'Minor Impact'
        elif s==3:
            severity = 'Major Impact'
        elif s==4:
            severity = 'Hazardous Impact'
        elif s==5:
            severity = 'Catastrophic Impact'
        curr_severities[hazard] = severity
    return curr_severities

def get_SAFENET_severity_USFS(severities):
    """
    Assigns a severity category according to USFS risk matrix

    Parameters
    ----------
    severities : dict
        Dictionary with keys as hazards and values as average severities.

    Returns
    -------
    curr_severities : dict
        Dictionary with keys as hazards and values as severity category.

    """
    curr_severities = {hazard:0 for hazard in severities}
    for hazard in severities:
        s = severities[hazard]
        if s== 1: #negligible impact
            severity = 'Negligible'
        elif s==2:
            severity = 'Marginal'
        elif s>2 and s<4:
            severity = 'Critical'
        elif s>=4:
            severity = 'Catastrophic'
        curr_severities[hazard] = severity
    return curr_severities