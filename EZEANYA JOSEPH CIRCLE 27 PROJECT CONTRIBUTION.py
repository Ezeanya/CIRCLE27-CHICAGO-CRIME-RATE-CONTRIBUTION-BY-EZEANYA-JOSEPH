#!/usr/bin/env python
# coding: utf-8

# In[1]:


# Importing the packages needed for analysis
import pandas as pd
import numpy as np
import datetime as dt
import matplotlib.pyplot as plt
import seaborn as sns

import warnings
warnings.filterwarnings('ignore')


# In[2]:


# Loading and reading my dataset
 

chicago_data = pd.read_csv(r'C:\Users\HP\Desktop\crime_data_chicago.csv')


#checking first five rows of my dataset.
chicago_data.head()


# In[3]:


# Checkking columns of the data set
chicago_data.columns


# In[4]:


# Checkking descriptions of our data set
chicago_data.describe()


# In[5]:


# Checkking shapes of our dataset
chicago_data.shape


# In[6]:


# Checkking data types of our data set
chicago_data.info()


# In[7]:


# converting date from object to datetime as date column can't have object type
chicago_data['Date'] = pd.to_datetime(chicago_data['Date'], format='%m/%d/%Y %I:%M:%S %p')

# Verify the data type
chicago_data.dtypes


# In[8]:


# Extract date columns
chicago_data['Month'] = chicago_data['Date'].dt.month
chicago_data['Day'] = chicago_data['Date'].dt.day


# In[9]:


# Dropping unneeded columns for analysis 
chicago_data.drop(['Unnamed: 0', 'Location', 'IUCR'], axis=1, inplace=True)

# change the dataFrame index to read good
chicago_data.index += 1

# Displaying the first few rows
chicago_data.head(10)


# In[10]:


# Checkking for missing values
chicago_data.isnull().sum()


# In[11]:


# Replace missing values in location description with NAN
chicago_data['Location Description'].fillna('NAN', inplace=True)


# In[12]:


# Replace missing values in district with the mode
chicago_data['District'].fillna(chicago_data['District'].mode()[0], inplace=True)


# In[13]:


# Replace missing values in Case Number with the mode
chicago_data['Case Number'].fillna(chicago_data['Case Number'].mode()[0], inplace=True)


# In[14]:


# Replace missing values in the ward column with the mode values
chicago_data['Ward'] = chicago_data.groupby('District')['Ward'].transform(lambda x: x.fillna(x.mode()[0]))
# Replace missing values in the community area column with the mode of the community area values
chicago_data['Community Area'] = chicago_data.groupby(['District','Ward'])['Community Area'].transform(lambda x: x.fillna(x.mode()[0]))


# In[15]:


# Replace missing values in the latitude and Longitude column with the median values
chicago_data['Latitude'] = chicago_data.groupby('District')['Latitude'].transform(lambda x: x.fillna(x.median()))
chicago_data['Longitude'] = chicago_data.groupby('District')['Longitude'].transform(lambda x: x.fillna(x.median()))


# In[16]:


# Imputing missing values in the X Coordinate and Y Coordinate column with the median  values
chicago_data['X Coordinate'] = chicago_data.groupby('District')['X Coordinate'].transform(lambda x: x.fillna(x.median()))
chicago_data['Y Coordinate'] = chicago_data.groupby('District')['Y Coordinate'].transform(lambda x: x.fillna(x.median()))


# In[17]:


# Confirm no missing values
chicago_data.isna().sum()


# In[18]:


# Check frequency of the distribution
chicago_data['Primary Type'].value_counts()


# In[19]:


#Most common days for theft
theft_df = chicago_data[chicago_data['Primary Type'] == 'THEFT']
theft_df.groupby('Day')['Primary Type'].count().sort_values(ascending=False)


# In[20]:


chicago_data.info()


# In[21]:


# List of Numeric Columns
numeric_columns = ['ID', 'District', 'Ward', 'Community Area', 'X Coordinate', 'Y Coordinate',
                   'Latitude', 'Longitude', 'Month', 'Day']

# Counting Numeric Values
total_numeric_values = sum(chicago_data[column].notnull().sum() for column in numeric_columns)

# Print the total number of numeric values
total_numeric_values


# In[22]:


# List of Numeric Columns
numeric_columns = ['ID', 'District', 'Ward', 'Community Area', 'X Coordinate', 'Y Coordinate',
                   'Latitude', 'Longitude', 'Month', 'Day']

# Create box plots for each numeric column
plt.figure(figsize=(14, 10))
for i, col in enumerate(numeric_columns, 1):
    plt.subplot(3, 4, i)
    sns.boxplot(x=chicago_data[col])
    plt.title(col)
plt.tight_layout()
plt.show()


# In[23]:


# Chekking number of arrest 
arrest_df = chicago_data[chicago_data['Arrest'] == True]

# arrests by month using groupby
arrest_by_month = arrest_df.groupby('Month')['Arrest'].count().sort_values(ascending=False).reset_index(name='Arrests')
arrest_by_month.style.background_gradient()


# In[24]:


# Create histograms for each numeric column
plt.figure(figsize=(12, 8))
for i, col in enumerate(numeric_columns, 1):
    plt.subplot(3, 4, i)
    plt.hist(chicago_data[col], bins=30, color='skyblue', edgecolor='black')
    plt.title(col)
plt.tight_layout()
plt.show()


# In[25]:


#plotting the number of arrests by month
plt.figure(figsize=(10, 6))
sns.barplot(x='Month', y='Arrests', data=arrest_by_month, palette='YlOrRd')
plt.title('Number of Arrests by Month')
plt.xlabel('Month')
plt.ylabel('Number of Arrests')
plt.xticks(rotation=45)
plt.tight_layout()
plt.show()


# In[26]:


# Creating a function to remove outliers
def remove_outliers(column):
    # Calculate Q1 and Q3
    Q1 = chicago_data[column].quantile(0.25)
    Q3 = chicago_data[column].quantile(0.75)
    
    # Calculate the IQR
    IQR = Q3 - Q1
    
    # Calculate the lower and upper bounds
    lower_bound = Q1 - 1.5 * IQR
    upper_bound = Q3 + 1.5 * IQR
    
    # removing outliers
    filtered_chicago_data = chicago_data[(chicago_data[column] >= lower_bound) & (chicago_data[column] <= upper_bound)]
    
    return filtered_chicago_data

# Remove outliers from each numeric column
for col in numeric_columns:
    chicago_data = remove_outliers(col)

# Display the DataFrame first five rows
chicago_data.head()


# In[27]:


#viewing the summary statistics (differential)
chicago_data.describe()


# In[43]:


# List of numeric columns
numeric_columns_1= ['ID', 'Beat', 'District', 'Ward', 'Community Area', 'X Coordinate', 'Y Coordinate', 'Year', 'Latitude', 'Longitude']

# Plot histograms for numeric variables
for column in numeric_columns_1:
    plt.figure(figsize=(7, 5))
    plt.hist(chicago_data[column], bins=30, color='red', edgecolor='blue')
    plt.title(f'Histogram of {column}')
    plt.xlabel(column)
    plt.ylabel('Frequency')
    plt.grid(True)
    plt.show()


# In[29]:



# visualizing primary crime types,

crime_types = chicago_data['Primary Type'].value_counts()

# Visualizing using a bar chart
plt.figure(figsize=(13, 8))
crime_types.plot(kind='bar')
plt.title('Distribution of Crime Types in Chicago')
plt.xlabel('Crime Type')
plt.ylabel('Number of Occurrences')
plt.xticks(rotation=90)
plt.show()


# In[30]:



# count the number of crimes by year
crime_trends = chicago_data.groupby('Year').size()

# Plot the crime trends
plt.figure(figsize=(12, 6))
crime_trends.plot(kind='line', marker='x', color='red')
plt.title('Crime Trends Over Time')
plt.xlabel('Year')
plt.ylabel('Number of Crimes')
plt.grid(True)
plt.show()


# In[31]:


# count the number of crimes each month
crime_trends_monthly = chicago_data.groupby('Month').size()

# Plot the crime trends by month
plt.figure(figsize=(12, 6))
crime_trends_monthly.plot(kind='line', marker='x', color='red')
plt.title('Crime Trends Over Time (Monthly)')
plt.xlabel('Month')
plt.ylabel('Number of Crimes')
plt.grid(True)
plt.show()


# In[32]:


# correlation chart to see the heat map
numeric_vars = chicago_data.select_dtypes(include='number')

# correlation matrix
correlation_matrix = numeric_vars.corr()

# Plotting correlation heatmap
plt.figure(figsize=(10, 6))
sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm', fmt=".2f", linewidths=.5)
plt.title('Correlation Heatmap')
plt.show()


# In[33]:


# Group the data by District and count the number of crimes in each district
crime_by_district = chicago_data['District'].value_counts()

# Plot the distribution of crimes by district
 # Rotate x-axis labels 
plt.figure(figsize=(12, 6))
crime_by_district.plot(kind='bar', color='skyblue')
plt.title('Number of Crimes per District')
plt.xlabel('District')
plt.ylabel('Number of Crimes')
plt.xticks(rotation=0) 
plt.tight_layout()
plt.show()


# In[34]:


# Plot the distribution of crimes by beat with rotated x-axis labels

crime_by_beat = chicago_data['Beat'].value_counts()


plt.figure(figsize=(12, 8))
crime_by_beat.plot(kind='bar', color='skyblue')
plt.title('Number of Crimes per Beat')
plt.xlabel('Beat')
plt.ylabel('Number of Crimes')
plt.xticks(rotation=45, ha='right')  
plt.tight_layout()
plt.show()


# In[35]:


# Filter the data for crimes that occurred in residences
residence_crimes = chicago_data[chicago_data['Location Description'] == 'RESIDENCE']

# count the number of crimes in each month
residence_crimes_by_month = residence_crimes.groupby('Month').size()

# Plotting trends of crime occurrences
plt.figure(figsize=(12, 6))
residence_crimes_by_month.plot(kind='line', marker='x', color='red')
plt.title('Monthly Trends of Crime Occurrences in Residences')
plt.xlabel('Month')
plt.ylabel('Number of Crimes')

# Customize the x-axis labels using the existing Month column
months = ['January', 'February', 'March', 'April', 'May', 'June', 'July', 'August', 'September', 'October', 'November', 'December']
plt.xticks(range(1, 13), months, rotation=45)

plt.grid(True)
plt.tight_layout()
plt.show()


# In[36]:


# Group the  data by year and count the number of crimes in each year
residence_crimes_by_year = residence_crimes.groupby('Year').size()

# Plot the yearly trends of crime occurrences in residences
plt.figure(figsize=(12, 6))
residence_crimes_by_year.plot(kind='line', marker='x', color='red')
plt.title('Yearly Trends of Crime Occurrences in Residences')
plt.xlabel('Year')
plt.ylabel('Number of Crimes')

plt.grid(True)
plt.tight_layout()
plt.show()


# In[37]:


chicago_data['Ward'].value_counts()

# Plotting the distribution across wards
plt.figure(figsize=(12, 8))
sns.countplot(x='Ward', data=chicago_data, order=chicago_data['Ward'].value_counts().index)
plt.title('Distribution of Crime Incidents Across Wards in Chicago')
plt.xlabel('Ward')
plt.ylabel('Number of Incidents')
plt.xticks(rotation=90)
plt.tight_layout()
plt.show()


# In[38]:


# Create a new column for weekday
chicago_data['Weekday'] = chicago_data['Day'].isin(['Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday'])


# Plotting the distribution of crime incidents between weekdays and weekends
plt.figure(figsize=(6, 5))
sns.countplot(x='Weekday', data=chicago_data, palette='Set2')
plt.title('Distribution of Crime Incidents Between Weekdays and Weekends')
plt.xlabel('Day of the Week')
plt.ylabel('Number of Incidents')
plt.xticks(ticks=[0, 1], labels=['Weekend', 'Weekday'])
plt.tight_layout()
plt.show()


# In[39]:


# Create a new column for weekend
chicago_data['Weekend'] = chicago_data['Day'].isin(['Saturday', 'Sunday'])

# Plotting the distribution of crime incidents between weekdays and weekends
plt.figure(figsize=(6, 5))
sns.countplot(x='Weekend', data=chicago_data, palette='Set2')
plt.title('Distribution of Crime Incidents Between Weekdays and Weekends')
plt.xlabel('Day of the Week')
plt.ylabel('Number of Incidents')
plt.xticks(ticks=[0, 1], labels=['Weekend', 'Weekday'])
plt.tight_layout()
plt.show()


# In[40]:


# Grouping crime primary type
domestic_crime_distribution = chicago_data.groupby('Domestic')['Primary Type'].value_counts().unstack().fillna(0)
domestic_crime_distribution

# Plotting a grouped bar chart
domestic_crime_distribution.plot(kind='bar', stacked=True, figsize=(10, 6))
plt.title('Distribution of Crime Types by Domestic vs. Non-Domestic Incidents')
plt.xlabel('Domestic')
plt.ylabel('Number of Incidents')
plt.xticks(rotation=0)
plt.legend(title='Primary Type', bbox_to_anchor=(1.05, 1), loc='upper left')
plt.tight_layout()
plt.show()


# In[41]:


# Chekking crimes that occurred in apartments
apartment_crimes = chicago_data[chicago_data['Location Description'] == 'APARTMENT']

# count the number of crimes in each year
apartment_crimes_by_year = apartment_crimes.resample('Y', on='Date').size()

# Plotting the yearly trends
plt.figure(figsize=(12, 6))
apartment_crimes_by_year.plot(kind='line', marker='x', color='red')
plt.title('Yearly Trends of Crime Occurrences in Apartments')
plt.xlabel('Year')
plt.ylabel('Number of Crimes')
plt.grid(True)
plt.tight_layout()
plt.show()


# In[42]:


#count the number of crimes in each district
crime_by_district = chicago_data['District'].value_counts()

# Plot the distribution of crimes by district

plt.figure(figsize=(12, 6))
crime_by_district.plot(kind='bar', color='red')
plt.title('Number of Crimes per District')
plt.xlabel('District')
plt.ylabel('Number of Crimes')
plt.xticks(rotation=0)  # Rotate x-axis labels for better readability
plt.tight_layout()
plt.show()


# In[ ]:




