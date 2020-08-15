import pandas as pd
import sys
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import KFold
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import classification_report
import copy
from sklearn.svm import SVC
import pickle
import matplotlib.pyplot as plt
import seaborn as sn
from sklearn.linear_model import LinearRegression
from collections import Counter


df = pd.read_csv("Provider_Info_original.csv")

#table for df describe
summary_statistics = df.describe().T
summary_statistics = summary_statistics.apply(lambda x: round(x,3))
summary_statistics = summary_statistics[3:]
summary_statistics = summary_statistics.sort_values(by=list(summary_statistics.columns)[1:], ascending=False)
#summary_statistics.to_csv("description_table.csv")
print(summary_statistics)



#remove target variables containing NaNs
df = df[df['Overall Rating'].notna()]


#remove all columns with constant values
cols_removable = []
for i in df.columns:
    try:
        if(df[i].max() == df[i].min()):
            cols_removable.append(i)
    except:
        continue


#Exploratory Data Analysis and Data Pre-Processing
df_new = df[['Location','Federal Provider Number','Provider State','Total Weighted Health Survey Score','Overall Rating','Total Amount of Fines in Dollars']]



cols_removable.extend(['Location','Rating Cycle 3 Standard Health Survey Date','Rating Cycle 2 Standard Health Survey Date','Rating Cycle 1 Standard Survey Health Date',
                        'Provider Name','Provider Address','Provider City','Provider State','Provider Zip Code','Provider Phone Number','Provider SSA County',
                        'Provider County Name','Ownership Type','Legal Business Name','Date First Approved to Provide Medicare and Medicaid services',
                        'Staffing Rating Footnote','RN Staffing Rating', 'RN Staffing Rating Footnote', 'Case-Mix Total Nurse Staffing Hours per Resident per Day',
                        'Case-Mix RN Staffing Hours per Resident per Day','Case-Mix LPN Staffing Hours per Resident per Day','Case-Mix Nurse Aide Staffing Hours per Resident per Day',
                        'Reported Physical Therapist Staffing Hours per Resident Per Day','Reported Total Nurse Staffing Hours per Resident per Day','Reported Licensed Staffing Hours per Resident per Day',
                        'Reported RN Staffing Hours per Resident per Day','Reported LPN Staffing Hours per Resident per Day','Reported Nurse Aide Staffing Hours per Resident per Day', 'Staffing Rating Footnote',
                        'Federal Provider Number'
])


df.drop(cols_removable, axis=1, inplace=True)
df.dropna(axis=1, how='all', inplace=True)

df["Provider Resides in Hospital"].replace({True:1,False:0}, inplace=True)
df["Continuing Care Retirement Community"].replace({True:1,False:0}, inplace=True)

df["Special Focus Status"].fillna(0, inplace=True)
df["Special Focus Status"] = pd.Series([1 for i in df["Special Focus Status"] if i!=0])

df["Abuse Icon"].replace({True:1,False:0}, inplace=True)

df["Most Recent Health Inspection More Than 2 Years Ago"].replace({True:1,False:0}, inplace=True)
df["Provider Changed Ownership in Last 12 Months"].replace({True:1,False:0}, inplace=True)

df["Automatic Sprinkler Systems in All Required Areas"].replace({"Yes":1,"Partial":0.5}, inplace=True)


df["QM Rating"].fillna(0,inplace=True)
df["Long-Stay QM Rating"].fillna(0, inplace=True)
df["Short-Stay QM Rating"].fillna(0, inplace=True)
df["Staffing Rating"].fillna(0,inplace=True)


df["Adjusted Nurse Aide Staffing Hours per Resident per Day"].fillna(df["Adjusted Nurse Aide Staffing Hours per Resident per Day"].mean(), inplace=True)
df["Adjusted LPN Staffing Hours per Resident per Day"].fillna(df["Adjusted LPN Staffing Hours per Resident per Day"].mean(), inplace=True)
df["Adjusted Total Nurse Staffing Hours per Resident per Day"].fillna(df["Adjusted Total Nurse Staffing Hours per Resident per Day"].mean(), inplace=True)
df["Adjusted RN Staffing Hours per Resident per Day"].fillna(df["Adjusted RN Staffing Hours per Resident per Day"].mean(), inplace=True)

df["Average Number of Residents Per Day"].fillna(df['Average Number of Residents Per Day'].mean(),inplace=True)

df["Rating Cycle 3 Total Number of Health Deficiencies"].replace(".",1, inplace=True)
df["Rating Cycle 3 Number of Standard Health Deficiencies"].replace(".",1, inplace=True)
df["Rating Cycle 3 Number of Health Revisits"].replace(".",1, inplace=True)
df["Rating Cycle 3 Health Deficiency Score"].replace(".",1, inplace=True)
df["Rating Cycle 3 Health Revisit Score"].replace(".",1, inplace=True)
df["Rating Cycle 3 Total Health Score"].replace(".",1, inplace=True)
df["Rating Cycle 3 Number of Complaint Health Deficiencies"].replace(".",1,inplace=True)

#dummy encoding for With a Resident and Family Council
df2 = pd.get_dummies(df["With a Resident and Family Council"])
df2.columns = ["With a Resident and Family Council "+i for i in df2.columns]

#merging dataframes together on axis=1 (column-wise)
df = pd.concat([df,df2], axis=1)
df.drop("With a Resident and Family Council", axis=1,  inplace=True)


df3 = pd.get_dummies(df["Provider Type"])
df3.columns = ["Provider Type "+i for i in df3.columns]

#merging dataframes together on axis=1 (column-wise)
df = pd.concat([df,df3], axis=1)
df.drop("Provider Type", axis=1,  inplace=True)


df["Special Focus Status"].fillna(0)

#df.to_csv("Provider_Info.csv",index=False)


y_class = df["Overall Rating"].astype(int)

df.drop('Overall Rating', 1, inplace=True)


#convert all column types to float
for i in df.columns:
    #df[i] = pd.to_numeric(df[i])
    df[i] = df[i].astype(float)





#-------------------------------------------- data exploration and analysis ---------------------------------------------



#Barplots -  Total Weighted Health Survey Score by state
health_score_by_state = df_new.groupby(['Provider State'])['Total Weighted Health Survey Score'].agg('sum').sort_values(ascending=False)
ax = health_score_by_state.plot.bar(x='Provider State', y='Total Weighted Health Survey Score', rot=90)
plt.xlabel("State")
plt.ylabel("Total Weighted Health Survey Score")
plt.title("Bar plot depicting Total Weighted Health Survey Score by State")
plt.show()


#heatmap for state vs overall rating (Tableau)
overall_rating_by_state = df_new.groupby(['Provider State'])['Overall Rating'].agg('sum')
#overall_rating_by_state.to_csv("tableau_data.csv")



#boxplots - amount of fines by state 

fines_by_state = list(df_new.groupby(['Provider State'])['Total Amount of Fines in Dollars'])
top_fines_by_state = df_new.groupby(['Provider State'])['Total Amount of Fines in Dollars'].agg('sum').sort_values(ascending=False)
top_fines_by_state = list(top_fines_by_state[0:5].index)


boxplot_lists = []
state_list = []

for i in range(len(fines_by_state)):
    if (fines_by_state[i][0] in top_fines_by_state):
        boxplot_lists.append(list(fines_by_state[i][1]))
        state_list.append(fines_by_state[i][0])


fig = plt.figure()
ax = fig.add_subplot(111)
bp = ax.boxplot(boxplot_lists)
ax.set_xticklabels(state_list)
plt.title("Box plots for Total Amount of Fines given to Nursing Homes in Dollars by State")
plt.ylabel("Total Amount of Fines in Dollars")
plt.xlabel("States")
plt.show()




#correlation Matrix - correlation plot
corrMatrix = df.corr().abs()
upperTraingular = corrMatrix.where(np.triu(np.ones(corrMatrix.shape), k=1).astype(np.bool))
highly_correlated_columns = [column for column in upperTraingular.columns if any(upperTraingular[column] > 0.7)]
print(highly_correlated_columns)

corr_matrix = df.corr()
corr_matrix = corr_matrix[highly_correlated_columns]
corr_matrix = corr_matrix.loc[highly_correlated_columns]


sn.heatmap(corr_matrix, annot=True)
plt.title("Heatmap showing the Pearson's Correlation Matrix for the features")
plt.show()




#scatterplot between Overall Rating and Total Number of Penalties
df_new["Overall Rating"] = df_new["Overall Rating"].astype(int)
health_score_by_rating = df_new.groupby(['Overall Rating'])['Total Weighted Health Survey Score'].agg('sum')
plt.scatter(x=health_score_by_rating.index, y=health_score_by_rating)
plt.title("Scatter plot between Overall Rating and Total Weighted Health Survey Score")
plt.ylabel("Total Weighted Health Survey Score")
plt.xlabel("Overall Rating")
plt.show()

#Note that a lower survey score corresponds to fewer deficiencies and revisits, and thus better performance
#on the health inspection domain. 
#https://www.cms.gov/Medicare/Provider-Enrollment-and-Certification/CertificationandComplianc/downloads/usersguide.pdf


plt.scatter(x=health_score_by_rating.index, y=health_score_by_rating)
plt.title("Regression Analysis for Overall Rating and Total Weighted Health Survey Score")
plt.ylabel("Total Weighted Health Survey Score")
plt.xlabel("Overall Rating")


X = np.array(health_score_by_rating.index.T).reshape(-1,1)
Y = np.array(health_score_by_rating)

# regression analyses 

reg = LinearRegression().fit(X, Y)
print(reg.score(X, Y))  #Rsquared
print(reg.coef_) 
plt.plot(X, reg.predict(X),color='r')
plt.show()




# hypothesis testing

from scipy.stats import chi2_contingency
csq1=chi2_contingency(pd.crosstab(df["Provider Resides in Hospital"], df["Continuing Care Retirement Community"]))
print("P-value: ",csq1[1])



csq2 = chi2_contingency(pd.crosstab(df["Provider Changed Ownership in Last 12 Months"], df["Abuse Icon"]))
print("P-value: ",csq2[1])




#normalize each column
for col in df.columns:
    df[col] = (df[col] - df[col].min()) / (df[col].max() - df[col].min())


df.dropna(axis=1, how='all', inplace=True)
#df.to_csv("Provider_Info_normalized.csv",index=False)


#class distribution
class_distribution = Counter(y_class)
bar = plt.bar(["1","2","3","4","5"], [class_distribution[1],class_distribution[2],class_distribution[3],class_distribution[4],class_distribution[5]])


bar[0].set_color("#2EC4B6")
bar[1].set_color("#E71D36")
bar[2].set_color("#ff9f1c")
bar[3].set_color("#7dce82")
bar[4].set_color("#db93b0")


plt.title("Class distribution")
plt.xlabel("Overall Rating (Class/Label)")
plt.ylabel("Number of Records")
plt.show()




#train-test split
train, test = train_test_split(df, test_size=0.2)

train_index = train.index
test_index = test.index

train_x_final = copy.deepcopy(train)
train_y_final = y_class[train_index]

test_x_final = copy.deepcopy(test)
test_y_final = y_class[test_index]


#uncomment to find the best hyper-parameters through GridSearchCSV
"""
print("Finding the best parameters for Random Forest ... ")
clf = RandomForestClassifier(random_state=0)
param_grid = { 
    'n_estimators': [500],
    'max_features': ['auto'],
    'max_depth' : [30],
    'criterion' :['entropy']
    }

random_forest_clf = GridSearchCV(estimator=clf, param_grid=param_grid, cv= 5, verbose=1)
random_forest_clf.fit(train_x_final, train_y_final)
print(random_forest_clf.best_params_)

#Outputs: {'criterion': 'entropy', 'max_depth': 30, 'max_features': 'auto', 'n_estimators': 500}
"""

"""
#svm grid search
print("Finding the best parameters for Support Vector Machine ... ")
param_grid = {'C': [10, 100], 
                 'gamma': [1,10]} 
# Make grid search classifier
clf_svm_grid = GridSearchCV(svm.SVC(), param_grid, verbose=1)
clf_svm_grid.fit(train_x_final, train_y_final)
clf_svm = clf_svm_grid.best_params_
print(clf_svm)

#Outputs: {'C': 10, 'gamma': 1}
"""


#K-Fold cross-validation
k_fold = KFold(n_splits=5, random_state=None)

for train_index, test_index in k_fold.split(train_x_final):

    print("*"*100)
    
    x_train_cv = train_x_final.iloc[train_index]
    x_test_cv  = train_x_final.iloc[test_index]
    y_train_cv = train_y_final.iloc[train_index]
    y_test_cv  = train_y_final.iloc[test_index]


    print("Training Random Forest ... ")
    rf = RandomForestClassifier(criterion="entropy", max_depth=30, max_features="auto", n_estimators=500)
    rf.fit(x_train_cv,y_train_cv)
    y_pred_cv = rf.predict(x_test_cv)

    print(classification_report(y_test_cv, y_pred_cv))

    print("################# SVM ###############")
    
    print("Training Support Vector Classifier ... ")
    clf_svm = SVC(C=10, gamma=1)
    clf_svm.fit(x_train_cv,y_train_cv)
    y_pred_cv_svm = clf_svm.predict(x_test_cv)
    print(classification_report(y_test_cv, y_pred_cv_svm))


#final training and testing process for Random Forest
rf = RandomForestClassifier(criterion="entropy", max_depth=30, max_features="auto", n_estimators=500)
print(rf)

clf_svm = SVC(C=10, gamma=1)
print(clf_svm)

rf.fit(train_x_final,train_y_final)

rf_filename = 'random_forest_model.sav'
pickle.dump(rf, open(rf_filename, 'wb'))

y_pred_final_rf = rf.predict(test_x_final)
print("Final Classification Report for RF")

print(classification_report(test_y_final, y_pred_final_rf))



#final training and testing process for Support Vector Machine
clf_svm.fit(train_x_final,train_y_final)

svm_filename = 'svm_model.sav'
pickle.dump(clf_svm, open(svm_filename, 'wb'))

svm_model_trained = pickle.load(open(svm_filename, 'rb'))
y_pred_final = svm_model_trained.predict(test_x_final)
print("Final Classification Report for SVM")
print(classification_report(test_y_final, y_pred_final))


#Extracting the Feature Importances from the trained model and displaying them.
print(rf.feature_importances_)

#feature_importances
feature_importances_dict = dict(zip(df.columns,rf.feature_importances_))

feature_importances_dict = {k: v for k, v in sorted(feature_importances_dict.items(), key=lambda item: item[1], reverse=True)}
#print(feature_importances_dict)


plt.bar(range(len(feature_importances_dict)), list(feature_importances_dict.values()), align='edge', width=0.5)
plt.xticks(range(len(feature_importances_dict)), list(feature_importances_dict.keys()))
plt.title("Features arranged in the order of their Importance")
plt.xlabel("Features")
plt.ylabel("Feature Importance")
plt.xticks(rotation=90)
plt.show()