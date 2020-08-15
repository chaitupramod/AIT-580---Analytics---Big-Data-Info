import pandas as pd
import numpy as np
import warnings
import sys
warnings.filterwarnings('ignore')

df = pd.read_csv("responses.csv")

#cleaning the columns
col_list = []
for col in df.columns:
    if("Unnamed" in col):
        col_list.append(df.loc[0,col])
    else:
        col_list.append(col+"_"+str(df.loc[0,col]))

df = df.iloc[1:]
df.columns = col_list


#Dropping columns containing all NaN values
df = df.dropna(axis=1, how='all')


#Make three columns of Q1 to single column by allocating 1,2 and 3
df["Q1_1.0"].fillna(0, inplace=True)
df["Q1.1_4.0"].fillna(0, inplace=True)
df["Q1.2_DL1"].fillna(0, inplace=True)

df["Q1_1.0"] = pd.to_numeric(df["Q1_1.0"])
df["Q1.1_4.0"].replace(1,2, inplace=True)
df["Q1.2_DL1"].replace('1',3, inplace=True)

#df["Q1.2_DL1"] = pd.to_numeric(df["Q1.2_DL1"])
df["Q1"] = df["Q1_1.0"] + df["Q1.1_4.0"] + df["Q1.2_DL1"]



df.drop(['Q1_1.0','Q1.1_4.0','Q1.2_DL1'], 1, inplace=True)


#Convert Q2 to to single column {0,1}
df["Q2_Male"].fillna(0, inplace=True)
df["Q2.1_Female"].fillna(0, inplace=True)
df["Q2_Male"].replace('1',1, inplace=True)
df["Q2.1_Female"].replace('1',2, inplace=True)
df["Q2"] = df["Q2_Male"] + df["Q2.1_Female"]
df.drop(['Q2_Male','Q2.1_Female'], 1, inplace=True)


#Replace rows 49 and 47 with the mode of Age column
df['Q3_Age (years)'].fillna(df['Q3_Age (years)'].mode()[0], inplace=True)

#Replace null value with mode of Height column
df['Q4_Height (Inches)'].fillna(df['Q4_Height (Inches)'].mode()[0], inplace=True)


#Replace NaN with Unknown String
df['Q5_Country of Citizenship'].fillna("Unknown", inplace=True)
df['Q6_Undergraduate Degree'].fillna("Unknown", inplace=True)
df['Q7_Expected Graduation date from Mason MS program?'].fillna("Unknown", inplace=True)


#merging Q8 into single column
df["Q8_Microsoft/Windows"].fillna(0, inplace=True)
df["Q8.1_Apple/MacBook"].fillna(0, inplace=True)
df["Q8_Microsoft/Windows"].replace(1,1, inplace=True)
df["Q8.1_Apple/MacBook"].replace('1',2, inplace=True)
df["Q8_Microsoft/Windows"] = pd.to_numeric(df["Q8_Microsoft/Windows"])
df["Q8"] = df["Q8_Microsoft/Windows"] + df["Q8.1_Apple/MacBook"]
df.drop(['Q8_Microsoft/Windows','Q8.1_Apple/MacBook'], 1, inplace=True)


#45 index - Since Q1 is 1 for the oncampus-student we replace the commuting with the mean the campus students (1,2 - 001 and 004 sections)
Q1_comm_list = df[df["Q1"].isin([1,2])]
Q1_comm_list["Q9_Commuting time (minutes) from home/work to campus for class?"] = pd.to_numeric(Q1_comm_list["Q9_Commuting time (minutes) from home/work to campus for class?"])
mean_comm_list = Q1_comm_list["Q9_Commuting time (minutes) from home/work to campus for class?"].mean()
df.loc[45,"Q9_Commuting time (minutes) from home/work to campus for class?"] = mean_comm_list

#49 index - Since he is a DL student, his commute time is replaced bythe mode of DL students' commuting time
Q1_comm_list = df[df["Q1"].isin([3])]
Q1_comm_list["Q9_Commuting time (minutes) from home/work to campus for class?"] = pd.to_numeric(Q1_comm_list["Q9_Commuting time (minutes) from home/work to campus for class?"])
mean_comm_list = Q1_comm_list["Q9_Commuting time (minutes) from home/work to campus for class?"].mean()
df.loc[47,"Q9_Commuting time (minutes) from home/work to campus for class?"] = mean_comm_list


#Merge Q10 columns into one
df["Q10_Yes, Full Time"].fillna(0, inplace=True)
df["Q10.1_Working, but not Full Time"].fillna(0, inplace=True)
df["Q10.2_Not Working while attending Mason"].fillna(0, inplace=True)
df["Q10_Yes, Full Time"].replace('1',1, inplace=True)
df["Q10.1_Working, but not Full Time"].replace('1',2, inplace=True)
df["Q10.2_Not Working while attending Mason"].replace('1',3, inplace=True)
df["Q10_Yes, Full Time"] = pd.to_numeric(df["Q10_Yes, Full Time"])
df["Q10"] = df["Q10_Yes, Full Time"]+df["Q10.1_Working, but not Full Time"]+df["Q10.2_Not Working while attending Mason"]
df.drop(['Q10_Yes, Full Time','Q10.1_Working, but not Full Time','Q10.2_Not Working while attending Mason'], 1, inplace=True)


#Merge Q11 columns
df["Q11_Little/none"].fillna(0, inplace=True)
df["Q11.1_Some familiarity"].fillna(0, inplace=True)
df["Q11.2_Average user"].fillna(0, inplace=True)
df["Q11.3_Frequent use for projects"].fillna(0, inplace=True)
df["Q11.4_Fluent/expert"].fillna(0, inplace=True)

df["Q11.3_Frequent use for projects"] = pd.to_numeric(df["Q11.3_Frequent use for projects"])

df["Q11_Little/none"].replace('1',1, inplace=True)
df["Q11.1_Some familiarity"].replace('1',2, inplace=True)
df["Q11.2_Average user"].replace('1',3, inplace=True)
df["Q11.3_Frequent use for projects"].replace(1,4, inplace=True)
df["Q11.4_Fluent/expert"].replace('1',5, inplace=True)

df["Q11"] =  df["Q11_Little/none"] + df["Q11.1_Some familiarity"] + df["Q11.2_Average user"] + df["Q11.3_Frequent use for projects"] + df["Q11.4_Fluent/expert"]

df.drop(['Q11_Little/none','Q11.1_Some familiarity','Q11.2_Average user','Q11.3_Frequent use for projects','Q11.4_Fluent/expert'], 1, inplace=True)


#Upper case all country names
df['Q5_Country of Citizenship'] = df['Q5_Country of Citizenship'].str.replace(' ', '')
df['Q5_Country of Citizenship'] = df['Q5_Country of Citizenship'].str.upper()

df['Q5_Country of Citizenship'] = df['Q5_Country of Citizenship'].replace(['US','UNITEDSTATESOFAMERICA', 'UNITEDSTATES'],'USA')
df['Q5_Country of Citizenship'] = df['Q5_Country of Citizenship'].replace(['INDIAN'],'INDIA')




#Q6_Undergraduate Degree

df['Q6_Undergraduate Degree'] = df['Q6_Undergraduate Degree'].replace(['CS',
                                                                        'BE in Computer Science',
                                                                        'Computer science',
                                                                        ' Computer science ',
                                                                        'B.Tech in CS',
                                                                        'CSE',
                                                                        'computerscience',
                                                                        'Computer science',
                                                                        'Computer science ',
                                                                        'computer science'],'Computer Science')


df['Q6_Undergraduate Degree'] = df['Q6_Undergraduate Degree'].replace(["Bachelor's of technology",
                                                                        'BACHELORS',
                                                                        'Bachelors',
                                                                        'Bachelors in Engineering ',
                                                                        'Bachelor of Engineering',
                                                                        ' Bachelor of Engineering',
                                                                        'Btech',
                                                                        'India',
                                                                        'Unknown',
                                                                        'BTech',
                                                                        'Engineering',
                                                                        'Bachelor of Technology'],'Uncertain Specialization')


df['Q6_Undergraduate Degree'] = df['Q6_Undergraduate Degree'].replace(["Bachelor of electronics",
                                                                        'BACHELORS',
                                                                        'Electronics ',
                                                                        'Electrical Engineering'],'Electronics')


df['Q6_Undergraduate Degree'] = df['Q6_Undergraduate Degree'].replace(["Bachelor of electronics",
                                                                        'BACHELORS',
                                                                        'Electronics ',
                                                                        'Electrical Engineering'],'Electronics')
                                
df['Q6_Undergraduate Degree'] = df['Q6_Undergraduate Degree'].replace(["Electronics&Communication",
                                                                        'Electronics and comm'],'Electronics and Communication')

df['Q6_Undergraduate Degree'] = df['Q6_Undergraduate Degree'].replace(["Mathematics ",],'Mathematics')

df['Q6_Undergraduate Degree'] = df['Q6_Undergraduate Degree'].replace(["electronics and media ",
                                                                        'Electronics and Media ',
                                                                        'electronics and media'],'Electronics and Media')

df['Q6_Undergraduate Degree'] = df['Q6_Undergraduate Degree'].replace(["Information System"],'Information Systems')

df['Q6_Undergraduate Degree'] = df['Q6_Undergraduate Degree'].replace(["Computer application",
                                                                        'Computer application '],'Computer Application')



df.to_csv("cleaned_responses.csv",index=False)







				




			

