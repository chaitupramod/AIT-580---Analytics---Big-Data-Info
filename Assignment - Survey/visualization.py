import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import sys

df = pd.read_csv("cleaned_responses.csv")

print(df.columns)

col_001 = "#227C9D"
col_004 = "#17C3B2"
col_dl = "#FFCB77"


df_001 = df[df["Q1"]==1]
df_004 = df[df["Q1"]==2]
df_dl = df[df["Q1"]==3]



#summary statistics
#print(df_001.describe().T)
#print(df_004.describe().T)
#print(df_dl.describe().T)



#Q1
bar = plt.bar(["001","004","DL"], [len(df_001),len(df_004),len(df_dl)])
bar[0].set_color(col_001)
bar[1].set_color(col_004)
bar[2].set_color(col_dl)
plt.title("Student distribution by Section")
plt.xlabel("Section")
plt.ylabel("Number of students")
plt.show()


#Q2 (Gender)
def get_gender_counts(df_arg):

    return(df_arg.groupby("Q2")["Q2"].count()[1], df_arg.groupby("Q2")["Q2"].count()[2])

male_count_001, female_count_001 = get_gender_counts(df_001)
male_count_004, female_count_004 = get_gender_counts(df_004)
male_count_dl, female_count_dl = get_gender_counts(df_dl)

male = [male_count_001,male_count_004,male_count_dl]
female = [female_count_001,female_count_004,female_count_dl]

index = ['001', '004', 'DL']

temp_df = pd.DataFrame({'Male': male,'Female': female}, index=index)
ax = temp_df.plot.bar(rot=0)
plt.title("Grouped Bar Chart representing Gender distribution")
plt.xlabel("Sections")
plt.ylabel("Count")

plt.show()


#Q3 (Q3_Age (years))
age_001 = df_001["Q3_Age (years)"].tolist()
age_004 = df_004["Q3_Age (years)"].tolist()
age_dl = df_dl["Q3_Age (years)"].tolist()

data = [age_001,age_004,age_dl]
fig = plt.figure(1, figsize=(9, 6))
ax = fig.add_subplot(111)
bp = ax.boxplot(data)
ax.set_xticklabels(['001', '004', 'DL'])
plt.title("Box plots for Age distributions")
plt.ylabel("Age")
plt.xlabel("Sections")
plt.show()


#Q4_Height (Inches)

height_001 = df_001["Q4_Height (Inches)"].tolist()
height_004 = df_004["Q4_Height (Inches)"].tolist()
height_dl = df_dl["Q4_Height (Inches)"].tolist()

data = [height_001,height_004,height_dl]
fig = plt.figure(1, figsize=(9, 6))
ax = fig.add_subplot(111)
bp = ax.boxplot(data)
ax.set_xticklabels(['001', '004', 'DL'])
plt.title("Box plots for Height distributions")
plt.ylabel("Height")
plt.xlabel("Sections")
plt.show()


#Q5_Country of Citizenship

def get_nationality_counts(df_arg):

    
    return(df_arg.groupby("Q5_Country of Citizenship")["Q5_Country of Citizenship"].count())

    #return(df_arg.groupby("Q5_Country of Citizenship")["Q5_Country of Citizenship"].count()[1], df_arg.groupby("Q2")["Q2"].count()[2])

countries = df["Q5_Country of Citizenship"].unique()

ndict_001 = get_nationality_counts(df_001).to_dict()
ndict_004 = get_nationality_counts(df_004).to_dict()
ndict_dl = get_nationality_counts(df_dl).to_dict()


list_dict = [ndict_001,ndict_004,ndict_dl]

from collections import defaultdict

final_dict = defaultdict(list)

for ndict in list_dict:
    for key, val in ndict.items():
        final_dict[key].append(val)

temp_dict = pd.DataFrame()
for key, value in final_dict.items():
    value.extend([0]*(3-len(value)))

temp_df = pd.DataFrame(final_dict,index = ['001', '004', 'DL'])

ax = temp_df.plot.bar(rot=0)
plt.title("Grouped Bar Chart representing Nationality distribution")
plt.xlabel("Sections")
plt.ylabel("Number of Students")
plt.show()


#Q7_Expected Graduation date from Mason MS program?

temp_dict = dict(df.groupby('Q1')['Q7_Expected Graduation date from Mason MS program?'].apply(list))
for key in temp_dict.keys():
    list_ = temp_dict[key]
    for j in range(0,len(list_)):
        try:
            trimmed_date = list_[j].split("-")
            list_[j] = trimmed_date[0]+"/"+trimmed_date[1]
        except:
            list_[j] = "Unknown"


dict_list = []

from collections import Counter
for key in temp_dict.keys():
    dict_list.append(Counter(temp_dict[key]))

final_dict = defaultdict(list)

for ndict in dict_list:
    for key, val in ndict.items():
        final_dict[key].append(val)

temp_dict = pd.DataFrame()
for key, value in final_dict.items():
    value.extend([0]*(3-len(value)))


temp_df = pd.DataFrame(final_dict,index = ['001', '004', 'DL'])

ax = temp_df.plot.bar(rot=0)
plt.title("Grouped Bar Chart representing Expected Month of Graduation")
plt.xlabel("Sections")
plt.ylabel("Number of Students")
plt.show()



#Q8

q8_001 = df_001.groupby("Q8")["Q8"].count().to_dict()
q8_004 = df_004.groupby("Q8")["Q8"].count().to_dict()
q8_dl = df_dl.groupby("Q8")["Q8"].count().to_dict()


list_0 = []
list_1 = []
list_2 = []
list_3 = []
q8_dicts = [q8_001,q8_004,q8_dl]
key_list=[0,1,2,3]

for dict_ in q8_dicts:
    for key in key_list:
            if key==0:
                try:
                    list_0.append(dict_[key])
                except:
                    list_0.append(0)
            if key==1:
                try:
                    list_1.append(dict_[key])
                except:
                    list_1.append(0)
            if key==2:
                try:
                    list_2.append(dict_[key])
                except:
                    list_2.append(0)
            if key==3:
                try:
                    list_3.append(dict_[key])
                except:
                    list_3.append(0)


#Nobody uses 'Other' option
data = [list_0,list_1,list_2,list_3]
temp_df = pd.DataFrame(data,index = ['Unknown','Microsoft/Windows','Apple/MacBook','Both'])

temp_df.columns = ["001","004","DL"]
print(temp_df)

temp_df.plot.pie(subplots=True, figsize=(30, 4), autopct='%1.2f%%', colors=["#91A8A4","#BAF2E9","#BAD7F2","#F2BAC9"])
plt.subplots_adjust(hspace=0.5)
plt.figtext(0.5,0.8, 'Distribution of type of Operating System used', fontsize=30, ha='center', va='center')
plt.show()


#Q9_Commuting time (minutes) from home/work to campus for class?
comm_001 = df_001["Q9_Commuting time (minutes) from home/work to campus for class?"].tolist()
comm_004 = df_004["Q9_Commuting time (minutes) from home/work to campus for class?"].tolist()
comm_dl = df_dl["Q9_Commuting time (minutes) from home/work to campus for class?"].tolist()

data = [comm_001,comm_004,comm_dl]
fig = plt.figure(1, figsize=(9, 6))
ax = fig.add_subplot(111)
bp = ax.boxplot(data)
ax.set_xticklabels(['001', '004', 'DL'])
plt.title("Box plots for Commuting time")
plt.ylabel("Commuting time")
plt.xlabel("Sections")
plt.show()


#Q10
q10_001 = df_001.groupby("Q10")["Q10"].count().to_dict()
q10_004 = df_004.groupby("Q10")["Q10"].count().to_dict()
q10_dl = df_dl.groupby("Q10")["Q10"].count().to_dict()


list_0 = []
list_1 = []
list_2 = []
list_3 = []
q10_dicts = [q10_001,q10_004,q10_dl]
key_list=[0,1,2,3]

for dict_ in q10_dicts:
    for key in key_list:
            if key==0:
                try:
                    list_0.append(dict_[key])
                except:
                    list_0.append(0)
            if key==1:
                try:
                    list_1.append(dict_[key])
                except:
                    list_1.append(0)
            if key==2:
                try:
                    list_2.append(dict_[key])
                except:
                    list_2.append(0)
            if key==3:
                try:
                    list_3.append(dict_[key])
                except:
                    list_3.append(0)


#Nobody uses 'Other' option
data = [list_0,list_1,list_2,list_3]
temp_df = pd.DataFrame(data,index = ["Unknown","Yes, Full Time", "Working, but not Full Time", "Not Working while attending Mason"])
temp_df.columns = ["001","004","DL"]
temp_df = temp_df.T

print(temp_df)
temp_df.iloc[:,:].plot.bar(stacked=True, color=["red","#91A8A4","#BAF2E9","#BAD7F2"], figsize=(10,7))
plt.title("Stacked bar graph representing employement status of students")
plt.ylabel("Number of Students")
plt.xlabel("Sections")
plt.show()


#Q11
q11_001 = df_001.groupby("Q11")["Q11"].count().to_dict()
q11_004 = df_004.groupby("Q11")["Q11"].count().to_dict()
q11_dl = df_dl.groupby("Q11")["Q11"].count().to_dict()

list_0 = []
list_1 = []
list_2 = []
list_3 = []
list_4 = []
list_5 = []


q11_dicts = [q11_001,q11_004,q11_dl]
key_list=[0,1,2,3,4,5]

for dict_ in q11_dicts:
    for key in key_list:
            if key==0:
                try:
                    list_0.append(dict_[key])
                except:
                    list_0.append(0)
            if key==1:
                try:
                    list_1.append(dict_[key])
                except:
                    list_1.append(0)
            if key==2:
                try:
                    list_2.append(dict_[key])
                except:
                    list_2.append(0)
            if key==3:
                try:
                    list_3.append(dict_[key])
                except:
                    list_3.append(0)
            if key==4:
                try:
                    list_4.append(dict_[key])
                except:
                    list_4.append(0)
            if key==5:
                try:
                    list_5.append(dict_[key])
                except:
                    list_5.append(0)



data = [list_0,list_1,list_2,list_3,list_4, list_5]
temp_df = pd.DataFrame(data,index = ["Unknown","Little/none", "Some familiarity", "Average user", "Frequent use for projects", "Fluent/expert"])
temp_df.columns = ["001","004","DL"]
temp_df = temp_df.T
temp_df.iloc[:,:].plot.bar(stacked=True, color=["#A4243B","#D8C99B","#D8973C","#BD632F","#273E47","#69DC9E"], figsize=(10,7))
plt.title("Stacked bar graph representing current level of programming skill in Python")
plt.ylabel("Number of Students")
plt.xlabel("Sections")
plt.show()


#------------------------------------------------------------------------------------

def get_spec_counts(df_arg):

    
    return(df_arg.groupby("Q6_Undergraduate Degree")["Q6_Undergraduate Degree"].count())

    #return(df_arg.groupby("Q5_Country of Citizenship")["Q5_Country of Citizenship"].count()[1], df_arg.groupby("Q2")["Q2"].count()[2])

spec = df["Q6_Undergraduate Degree"].unique()

spec_001 = get_spec_counts(df_001).to_dict()
spec_004 = get_spec_counts(df_004).to_dict()
spec_dl = get_spec_counts(df_dl).to_dict()


list_dict = [spec_001,spec_004,spec_dl]

from collections import defaultdict

final_dict = defaultdict(list)

for ndict in list_dict:
    for key, val in ndict.items():
        final_dict[key].append(val)

temp_dict = pd.DataFrame()
for key, value in final_dict.items():
    value.extend([0]*(3-len(value)))

temp_df = pd.DataFrame(final_dict,index = ['001', '004', 'DL'])

ax = temp_df.plot.bar(rot=0,color = ['#bada55','#7fe5f0','#ff0000','#ff80ed','#696969','#133337','#f7347a','#5ac18e','#065535','#ffc0cb','#008080',
                                      '#ffd700','#ff7373','#fa8072','#ffb6c1','#660066','#008000','#0e2f44','#088da5','#f6546a','#f08080','#0000ff',
                                      '#000000','#420420'])
plt.title("Grouped Bar Chart representing degree specialization distribution")
plt.xlabel("Sections")
plt.ylabel("Number of Students")
plt.show()