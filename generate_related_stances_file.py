#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Nov  6 12:51:31 2017

@author: mike
"""
import csv

with open('./data/competition_test_stances.csv') as inFile:
    with open('./data/comp_test_stances_related_only.csv', 'w+') as outfile:
        reader = csv.DictReader(inFile, delimiter=',', quotechar = '"')
        writer = csv.DictWriter(outfile, ['Headline','Body ID', 'Stance'])
        for row in reader:
            if row['Stance'] != 'unrelated':
                writer.writerow(row)
       
            
