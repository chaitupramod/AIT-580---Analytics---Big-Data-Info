# -*- coding: utf-8 -*-
"""
Created on Sun Mar  8 12:47:36 2020

@author: chait
"""

import pandas as pd
import matplotlib.pyplot as plt
import sys
from collections import Counter
import itertools

#a - Identify the analytical data type (NOIR) of each data item; explain your reasons.  Note that States_Affected is a compound data item.

#--------------------------------------------------------------------------------------------------------------

#b  - Load the dataset into Python; display a few records.

df = pd.read_csv("hurricanes.csv",delimiter = "|")
print(df.head())
print(df.tail())
print()
print()


#-------------------------------------------------------------------------------------------------------------
#j Many records have missing values; explain what you did to address that problem.

print("columns containing null vaules")
print(df.columns[df.isnull().any()])


print("Number of rows before dropping the column")
print(len(df))
df = df.drop(['Name'],axis=1)


df["Central_Pressure_mb"] = df["Central_Pressure_mb"].fillna(df["Central_Pressure_mb"].mode()[0])
df["Max_Winds_kt"] = df["Max_Winds_kt"].fillna(df["Max_Winds_kt"].mode()[0])


df["Highest_Category"] = df["Highest_Category"].str.replace('TS','0')
df["Highest_Category"] = df["Highest_Category"].astype(int)


print("Number of rows after dropping the column")
print(len(df))




#-------------------------------------------------------------------------------------------------------------
#c - Create summary tables and appropriate plots for Month, and for Highest_Category.  (Note: TS < Cat 1).


print("*****************************8")

#summary tables for Month and Highest Accuracy 
print("summary tables for Month")
print(df["Month"].describe())


print("summary tables for Highest Accuracy")
print(df["Highest_Category"].describe())



#bar plot for Month
count_df = df['Month'].value_counts()
bar_df = pd.DataFrame({'Month':list(count_df.index), 'Frequency':list(count_df.values)})

ax = bar_df.plot.bar(x='Month', y='Frequency', rot=0, title="Plot for Months vs Frequency")
ax.set_xlabel("Month")
ax.set_ylabel("Frequency")

#bar plot for highest category
count_df = df['Highest_Category'].value_counts()
bar_df = pd.DataFrame({'Month':list(count_df.index), 'Frequency':list(count_df.values)})
ax = bar_df.plot.bar(x='Month', y='Frequency', rot=0, title="Plot for values in Highest_Category column vs Frequency")
ax.set_xlabel("Highest Category")
ax.set_ylabel("Frequency")

#-------------------------------------------------------------------------------------------------------------
#d - Is there a relationship between Central_Pressure_mb and Max_Winds_kt? Explain your analysis and answer.


plt.figure()
plt.scatter(df["Central_Pressure_mb"], df["Max_Winds_kt"])
plt.xlabel("Highest Category")
plt.ylabel("Max_Winds_kt")
plt.show()

print("Pearson Correlation Coefficient Value between Central_Pressure_mb and Max_Winds_kt")
print(df['Central_Pressure_mb'].corr(df['Max_Winds_kt'])) 




#-------------------------------------------------------------------------------------------------------------
#e - Is there a relationship between Highest_Category and Central_Pressure_mb? Explain your analysis and answer.


df["Highest_Category"] = df["Highest_Category"].replace('TS',0)
df["Highest_Category"] = df["Highest_Category"].astype(int)

print("Pearson Correlation Coefficient Value between Highest_Category and Central_Pressure_mb")
print(df['Highest_Category'].corr(df['Central_Pressure_mb']))  #weak in the inverse direction

plt.figure()

plt.scatter(df["Highest_Category"], df["Central_Pressure_mb"]) #0 is TS
plt.xlabel("Highest_Category")
plt.ylabel("Central_Pressure_mb")
plt.show()
         
      
#-------------------------------------------------------------------------------------------------------------
#f - Display a table and visualization of Months. Explain the results

count_df = df['Month'].value_counts()
bar_df = pd.DataFrame({'Month':list(count_df.index), 'Frequency':list(count_df.values)})

print(bar_df)

ax = bar_df.plot.bar(x='Month', y='Frequency', rot=0, title="Frequency of Months")
ax.set_xlabel("Month")
ax.set_ylabel("Frequency")



#-------------------------------------------------------------------------------------------------------------
#g - Parse and summarize the data in States_Affected; explain your method and results.  Challenging!



count_dict = {}

import numpy as np


def states_affected(x):
    
    for j in x.split(";"):
        
        word_split = j.split(",")

        for k in range(1,len(word_split)):
            
            val = list(filter(str.isdigit,word_split[k]))
            
            key = word_split[0]+" "+word_split[k].strip(val[0])
            
            if key in count_dict.keys():
                count_dict[key] = list(np.append(count_dict[key],val))
                
            else:
                
                count_dict[key] = val
                
        return(count_dict)
                

            

df["States_Affected"] = df["States_Affected"].str.replace('TS','0')
df["States_Affected"].apply(states_affected)

count_states = []

for key in count_dict.keys():
    count_states.append(Counter(count_dict[key]))
  
    
count_states_df = pd.DataFrame(count_states)
count_states_df["States"] = count_dict.keys()
count_states_df = count_states_df.fillna(0)

print("Parsed and Summarized the data in States_Affected")
print()
print(count_states_df)
print(count_states_df.describe())


#-------------------------------------------------------------------------------------------------------------

#Create a table and a visualization showing the number of storms per year for each category. Be creative!

storm_count_by_year = {}
storm_count_by_year_list = []

groups = df.groupby("Year")

for name, group in groups:
   
    temp_df = group

    alpha_numeric = ''.join(list(temp_df["States_Affected"]))
    
    extracted_digits = ' '.join(filter(lambda i: i.isdigit(), alpha_numeric))
     
    storm_count_by_year[name] = extracted_digits.split(" ")
    
    
for key in storm_count_by_year.keys():
    storm_count_by_year_list.append(Counter(storm_count_by_year[key]))


print("Number of storms per year for each category")

storm_count_by_year_df = pd.DataFrame(storm_count_by_year_list)
storm_count_by_year_df["Year"] = storm_count_by_year.keys()
storm_count_by_year_df = storm_count_by_year_df.fillna(0)

print(storm_count_by_year_df)

storm_count_by_year_df[['1','2','3','4','5']].plot(kind='barh', stacked=True)
plt.ylabel("Year")
plt.xlabel("Number of storms per year")



#-------------------------------------------------------------------------------------------------------------
#Create a table and a visualization showing the number of storms per state for each category. Be creative!

print("Table showing the number of storms per state for each category")
print(count_states_df)



print("Visualization showing the number of storms per state for each category")


count_states_df.set_index('States', inplace=True)
count_states_df[['1','2','3','4','5']].plot(kind='bar', stacked=True)




    

    




















