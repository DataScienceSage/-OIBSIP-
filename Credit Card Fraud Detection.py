#!/usr/bin/env python
# coding: utf-8

# ## Importing Libraries

# In[147]:


import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px
import plotly.graph_objects as go
import xgboost as xgb
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix


# In[5]:


df = pd.read_csv('creditcard.csv')


# In[6]:


#Top 5 rows 
df.head()


# In[7]:


#Last 5 rows
df.tail()


# In[12]:


# Data Inforamtion
df.info()


# In[32]:


df.columns


# In[15]:


# Verifying if there are any null values 
df.isna().values.any()


# ## Statistics on the amounts

# In[18]:


# Statistics on the amounts
df.Amount.describe().round(2)


# In[31]:


# Distribution of Amounts
fig = px.scatter(df, x = 'Amount', y =df.index, color = df.Class,
                title = 'Distribution of Amount Values')
fig.update_layout(xaxis_title='Transaction Amount',
                    yaxis_title='Transactions')
fig.show('png')


# ### Class Distribution

# In[19]:


df['Class'].value_counts()


# ### 0 = Normal Transaction

# ### 1 = fraudulent transaction

# In[22]:


# separating the data for analysis
legit = df[df.Class == 0]
fraud = df[df.Class == 1]


# In[42]:


#printing the values
print(f"Shape of Legit transactions: {legit.shape}")
print(f"Shape of Fraudulant transactions: {fraud.shape}")


# In[118]:


# Visualizing the distribution of transaction classes as a pie chart
count_classes = df['Class'].value_counts().reset_index()
count_classes.columns = ['Class', 'Count']

fig = px.pie(
    count_classes, 
    names='Class', 
    values='Count', 
    title='Transaction Class Distribution', 
    color='Class',
    color_discrete_map={0: 'green', 1: 'red'}  
)

fig.show('png')


# In[68]:


# statistical measures of the data
legit_stats = legit.Amount.describe().round(2)

# Print summary statistics
print("Legit Transactions Statistics:")
print(legit_stats)


# In[69]:


fraud_stats = fraud.Amount.describe().round(2)

print("\nFraudulent Transactions Statistics:")
print(fraud_stats)


# In[64]:


# Create a DataFrame for easy plotting
summary_stats = pd.DataFrame({
    'Statistic': legit_stats.index,
    'Legit Transactions': legit_stats.values,
    'Fraudulent Transactions': fraud_stats.values
})

# Create the bar chart
fig = go.Figure()

# Add Legit Transactions bar
fig.add_trace(go.Bar(
    x=summary_stats['Statistic'],
    y=summary_stats['Legit Transactions'],
    name='Legit Transactions',
    marker_color='#1f77b4'  
))

# Add Fraudulent Transactions bar
fig.add_trace(go.Bar(
    x=summary_stats['Statistic'],
    y=summary_stats['Fraudulent Transactions'],
    name='Fraudulent Transactions',
    marker_color='#ff7f0e' 
))

# Update the layout
fig.update_layout(
    title='Summary Statistics of Transaction Amounts',
    xaxis_title='Statistic',
    yaxis_title='Amount',
    barmode='group'
)

# Show the figure
fig.show('png')


# In[27]:


# compare the values for both transactions
df.groupby('Class').mean()


# ### No. of Fraudulent Transactions = 492

# In[70]:


legit_sample = legit.sample(n=492)


# ### Concatenating two DataFrames

# In[72]:


new_df = pd.concat([legit_sample, fraud], axis=0)
new_df.head()


# In[73]:


new_df['Class'].value_counts()


# In[ ]:


#DRAW A GRAPH


# In[74]:


new_df.groupby('Class').mean()


# ### Splitting the data into Features & Targets

# In[75]:


X = new_df.drop(columns='Class', axis=1)
Y = new_df['Class']


# In[76]:


print(X)


# In[77]:


print(Y)


# ### Split the data into Training data & Testing Data

# In[80]:


X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2, stratify=Y, random_state=2)

print(X.shape, X_train.shape, X_test.shape)


# ### Logistic Regression

# In[82]:


model = LogisticRegression()


# In[83]:


# training the Logistic Regression Model with Training Data
model.fit(X_train, Y_train)


# ### Evaluation Model

# In[85]:


#Accuracy Score on training data
X_train_prediction = model.predict(X_train)
training_data_accuracy = accuracy_score(X_train_prediction, Y_train)

print('Accuracy on Training data : ', training_data_accuracy)


# In[86]:


# accuracy on test data
X_test_prediction = model.predict(X_test)
test_data_accuracy = accuracy_score(X_test_prediction, Y_test)

print('Accuracy score on Test Data : ', test_data_accuracy)


# In[131]:


# Generate predictions
y_pred = model.predict(X_test)

# Compute confusion matrix
cm = confusion_matrix(Y_test, y_pred)

# Plot confusion matrix
plt.figure(figsize=(10, 7))
sns.heatmap(cm, annot=True, fmt='d', cmap='Purples', xticklabels=['Class 0', 'Class 1'], yticklabels=['Class 0', 'Class 1'])
plt.xlabel('Predicted')
plt.ylabel('True')
plt.title('Confusion Matrix')
plt.show()


# ### Random Forest

# In[90]:


# Initialize and train the Random Forest model
rf_model = RandomForestClassifier(n_estimators=100, random_state=42)
rf_model.fit(X_train, Y_train)


# In[91]:


# Make predictions
rf_predictions = rf_model.predict(X_test)


# In[93]:


# Evaluate the model
rf_accuracy = accuracy_score(Y_test, rf_predictions)
print(f"Random Forest Accuracy: {rf_accuracy:.4f}")


# In[97]:


# Confusion Matrix
rf_conf_matrix = confusion_matrix(Y_test, rf_predictions)
print("Confusion Matrix:\n", rf_conf_matrix)


# In[98]:


# Classification Report
rf_class_report = classification_report(Y_test, rf_predictions)
print("Classification Report:\n", rf_class_report)


# In[99]:


# Plot Confusion Matrix
plt.figure(figsize=(8, 6))
sns.heatmap(rf_conf_matrix, annot=True, fmt='d', cmap='Blues', 
            xticklabels=['Normal', 'Fraudulent'], 
            yticklabels=['Normal', 'Fraudulent'])
plt.xlabel('Predicted')
plt.ylabel('Actual')
plt.title('Random Forest Confusion Matrix')
plt.show()


# ### XGBoost

# In[105]:


# Initialize and train the XGBoost model
xgb_model = xgb.XGBClassifier(use_label_encoder=False, eval_metric='logloss', random_state=42)
xgb_model.fit(X_train, Y_train)


# In[106]:


# Make predictions
xgb_predictions = xgb_model.predict(X_test)


# In[108]:


# Evaluate the model
xgb_accuracy = accuracy_score(Y_test, xgb_predictions)
print(f"XGBoost Accuracy: {xgb_accuracy:.4f}")


# In[109]:


# Confusion Matrix
xgb_conf_matrix = confusion_matrix(Y_test, xgb_predictions)
print("Confusion Matrix:\n", xgb_conf_matrix)


# In[110]:


# Classification Report
xgb_class_report = classification_report(Y_test, xgb_predictions)
print("Classification Report:\n", xgb_class_report)


# In[119]:


# Plot Confusion Matrix
plt.figure(figsize=(8, 6))
sns.heatmap(xgb_conf_matrix, annot=True, fmt='d', cmap='plasma', 
            xticklabels=['Normal', 'Fraudulent'], 
            yticklabels=['Normal', 'Fraudulent'])
plt.xlabel('Predicted')
plt.ylabel('Actual')
plt.title('XG')


# ### Comparing the results of models

# In[143]:


print("Logistic Regression Accuracy:", log_reg_accuracy)
print("Random Forest Accuracy:", rf_clf_accuracy)
print("XGBoost Accuracy:", xgb_clf_accuracy)


# In[146]:


import matplotlib.pyplot as plt

# Accuracy values
accuracies = [log_reg_accuracy, rf_clf_accuracy, xgb_clf_accuracy]
models = ['Logistic Regression', 'Random Forest', 'XGBoost']

# Create the bar chart
plt.figure(figsize=(10, 6))
bars = plt.bar(models, accuracies, color=['blue', 'green', 'red'])

# Add accuracy labels on the bars
for bar in bars:
    yval = bar.get_height()
    plt.text(bar.get_x() + bar.get_width()/2, yval, f'{yval:.4f}', va='bottom')  # va: vertical alignment

# Enhance the plot
plt.xlabel('Model')
plt.ylabel('Accuracy')
plt.title('Model Accuracy Comparison')
plt.ylim(0, 1)  # Accuracy ranges from 0 to 1
plt.grid(axis='y', linestyle='--', alpha=0.7)
plt.show()


# In[ ]:




