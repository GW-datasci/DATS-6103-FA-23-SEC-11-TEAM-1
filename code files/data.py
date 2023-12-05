
#%%
import seaborn as sns
import matplotlib.pyplot as plt
import pandas as pd
import missingno as msno


#%%
df = pd.read_csv('shopping_trends.csv')
df

# %%
# Summary of the data set
df.describe()

# %%
df.isnull().sum()

# %%
# Create a heatmap to visualize missing values
plt.figure(figsize=(10, 6))
sns.heatmap(df.isnull(), cbar=False, cmap='viridis')
plt.title('Missing Values in the Dataset')
plt.show()

print("From the above plot, we see that the dataset has no missing values.")

#%%
plt.figure(figsize = (10, 6))
ax = df["Gender"].value_counts().plot(kind = 'bar', color = ["#7f8c8d","#f39c12"], rot = 0)
ax.set_xticklabels(('Male', 'Female'))
plt.title("Gender Distribution",weight="bold")
plt.xlabel('Gender', fontsize = 14, labelpad = 20)
plt.ylabel('Count', fontsize = 14, labelpad = 20)

#%%
fig, ax = plt.subplots(figsize = (10, 6))
ax.hist(df['Age'], bins = 25, edgecolor = 'black', alpha = 0.7, color = '#1f77b4', density = True)
df['Age'].plot(kind = 'kde', color = 'red', ax = ax)
ax.set_xlabel('Age')
ax.set_ylabel('Count / Density')
ax.set_title('Age Distribution Histogram with Density Curve')
ax.legend(['Density Curve', 'Histogram'])
plt.show()

#%%
plt.figure(figsize = (12,8))
colors = ['#3498db', '#2ecc71', '#e74c3c', '#f1c40f']
ax = df["Category"].value_counts().plot(kind = 'bar', color=colors, rot = 0)
ax.set_xticklabels(('Clothing', 'Accessories', 'Footwear', 'Outerwear'))
plt.title("Category")
for p in ax.patches:
    ax.annotate(int(p.get_height()), (p.get_x() + 0.25, p.get_height() + 1), ha = 'center', va = 'bottom', color = 'black')
    ax.tick_params(axis = 'both', labelsize = 15)
plt.xlabel('Product Category', fontsize = 14, labelpad = 20)
plt.title("Distribution of Product Categories", weight = "bold", fontsize = 16)
plt.ylabel('Count', fontsize = 14, labelpad = 20)

#%%
plt.figure(figsize=(14, 8))
palette = sns.color_palette("viridis", n_colors=10)  # A gradient color palette
df["Location"].value_counts()[:10].sort_values(ascending=False).plot(kind='bar', color=palette, edgecolor='black')
plt.xlabel('State', weight="bold", fontsize=16, labelpad=20)
plt.ylabel('Frequency Count', weight="bold", fontsize=16, labelpad=20)
plt.xticks(rotation=45, ha='right', fontsize=14)  # Rotating for better label visibility
plt.yticks(fontsize=14)
plt.title("Top 10 States by Purchase Frequency", weight="bold", fontsize=18)
plt.tight_layout()
plt.show()

#%%
plt.figure(figsize = (10, 6))
colors = ['#3498db', '#2ecc71', '#e74c3c', '#f1c40f']
ax = df["Size"].value_counts().plot(kind = 'bar', color = colors, rot = 0)
ax.set_xticklabels(('Medium', 'Large', 'Small', 'Extra Large'))
plt.title("Clothing Size Distribution", weight='bold', fontsize='16')
for p in ax.patches:
    ax.annotate(int(p.get_height()), (p.get_x() + 0.25, p.get_height() + 1), ha = 'center', va = 'bottom', color = 'black')
    ax.tick_params(axis = 'both', labelsize = 15)
plt.xlabel('Size(s)', weight = "bold", fontsize = 14, labelpad = 20)
plt.ylabel('Count', weight = "bold", fontsize = 14, labelpad = 20)

#%%
plt.figure(figsize=(16, 8))
palette = sns.color_palette("terrain", n_colors=10)
df["Color"].value_counts()[:10].sort_values(ascending=True).plot(kind='barh', color=palette, edgecolor='black')
plt.xlabel('Frequency Count', weight="bold", fontsize=16, labelpad=20)
plt.ylabel('Color', weight="bold", fontsize=16, labelpad=20)
plt.xticks(fontsize=14)
plt.yticks(fontsize=14, rotation=0, ha='right')
plt.title("Top 10 Colors by Frequency", weight="bold", fontsize=18)
plt.tight_layout()
plt.show()

#%%
plt.figure(figsize=(16, 6))
counts = df["Season"].value_counts()
explode = (0, 0, 0, 0) 
colors = sns.color_palette("pastel")
counts.plot(kind='pie', fontsize=14, colors=colors, explode=explode, autopct='%1.1f%%', startangle=140)
plt.title("Distribution by Season", weight="bold", fontsize=16)
plt.axis('equal')
plt.legend(labels=counts.index, title="Seasons", loc="best", fontsize=12)
plt.show()

#%%
plt.figure(figsize = (10, 6))
ax = df["Subscription Status"].value_counts().plot(kind = 'bar', color =["red","blue"] , rot = 0)
ax.set_xticklabels(('No', 'Yes'))
plt.title("Subscription Status", weight='bold', fontsize = 16)
for p in ax.patches:
    ax.annotate(int(p.get_height()), (p.get_x() + 0.25, p.get_height() + 1), ha = 'center', va = 'bottom', color = 'black')
    ax.tick_params(axis = 'both', labelsize = 15)
plt.xlabel('Subscribed?', fontsize = 14, labelpad = 20)
plt.ylabel('Number of People',fontsize = 14, labelpad = 20)

#%%
plt.figure(figsize=(15, 6))
counts = df["Payment Method"].value_counts()
colors = sns.color_palette("Set2")
plt.title("Distribution of Payment Methods", weight="bold", fontsize=16)
counts.plot(kind='pie', fontsize=14, colors=colors, autopct='%1.1f%%', startangle=140)
plt.axis('equal')
plt.legend(labels=counts.index, loc="best", fontsize=12)
plt.show()

# EDA CODE BELOW ALSO USED IN SMART QUESTION.
# REMOVING EDA FROM HERE TO PREVENT REDUNDANCY.
# #%%
# plt.figure(figsize = (15, 6))
# counts = df["Frequency of Purchases"].value_counts()
# colors = sns.color_palette("tab10")
# plt.title("Frequency of Purchases Distribution", weight="bold", fontsize=16)
# counts.plot(kind='pie', fontsize=14, colors=colors, autopct='%1.1f%%', startangle=140)
# plt.axis('equal')
# plt.legend(labels=counts.index, loc="best", fontsize=12)
# plt.show()

# %% 
# We already did this at the beginning. Maybe remove the below code?
# Visualize the missing values using a matrix
# msno.matrix(df)
# plt.title('Missing Values Matrix')
# plt.show()
# Count missing values per column
# missing_counts = df.isnull().sum()
# plt.figure(figsize=(12, 6))
# sns.barplot(x=missing_counts.index, y=missing_counts.values)
# plt.xlabel('Columns')
# plt.ylabel('Number of Missing Values')
# plt.title('Count of Missing Values per Column')
# plt.xticks(rotation=90)
# plt.show()

# %%
# Create a scatterplot to visualize the relationship between customer age and purchase amount
sns.scatterplot(x='Age', y='Purchase Amount (USD)', data=df)
plt.xlabel('Customer Age')
plt.ylabel('Purchase Amount (USD)')
plt.title('Customer Age vs. Purchase Amount')
plt.show()

# %%
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
import pandas as pd

X = df['Age'].values.reshape(-1, 1)  # Predictor variable
y = df['Purchase Amount (USD)'].values  # Response variable

# create plot
plt.figure(figsize=(10, 6))

# scatterplot
plt.scatter(X, y, color='blue', label='Actual Data')

# create and bulid linearRegression
model = LinearRegression()
model.fit(X, y)
slope = model.coef_[0]
intercept = model.intercept_
# using model to predict
y_pred = model.predict(X)
plt.plot(X, y_pred, color='red', label='Regression Line')

plt.xlabel('Customer Age')
plt.ylabel('Purchase Amount (USD)')
plt.title('Linear Regression: Customer Age vs. Purchase Amount')
plt.legend()

plt.show()

print("Slope:", slope)
print("Intercept:", intercept)

#The red line in the graph indicates the line of best fit to the data. It indicates the general trend of the purchase amount as the customer's age changes.
#The slope of the regression line is approximately -0.0162, which means that for each additional year of age, the purchase amount decreases by approximately $0.016, with all other factors remaining constant.
#The intercept is approximately $60.48. Theoretically, this suggests that a customer who is zero years old would correlate to a purchase of approximately $60.48, it's not meaningful because the baby can't buy a car. 
#we can see the slope is very small, indicating a very weak negative correlation between age and purchase amount.

#Do correlation Test 
from scipy.stats import pearsonr

# Calculate Pearson's correlation coefficient and p-value
correlation, p_value = pearsonr(df['Age'], df['Purchase Amount (USD)'])

print(f"Pearson's Correlation: {correlation:.4f}")
print(f"P-Value: {p_value:.4f}")

# Interpret the results based on the p-value and correlation coefficient
if p_value < 0.05:
    print("There is a significant correlation between Age and Purchase Amount.")
else:
    print("There is no significant correlation between Age and Purchase Amount.")

#The correlation coefficient is approximately -0.0104, so indicating a very weak
#negative correlation between a customer's age and the purchase amount. And the p-value is 0.5152, which is greater than the typical significance level of 0.05.
#Therefore, you conclude that there is no significant correlation between  customer's age and the purchase amount. 

#%%
# Create a bar chart to compare purchase amounts by item category
sns.barplot(x='Category', y='Purchase Amount (USD)', data=df)
plt.xlabel('Item Category')
plt.ylabel('Average Purchase Amount (USD)')
plt.title('Item Category vs. Purchase Amount')
plt.xticks(rotation=45)
plt.show()

# %%
# Create a bar chart to compare purchase amounts by location
plt.figure(figsize=(12, 6))  # Adjust the figure size to fit the labels
ax = sns.barplot(x='Location', y='Purchase Amount (USD)', data=df)
plt.xlabel('Location')
plt.ylabel('Average Purchase Amount (USD)')
plt.title('Location vs. Purchase Amount')
ax.set_xticklabels(ax.get_xticklabels(), rotation=90, fontsize=10)  # Adjust rotation and label size
plt.show()

# %%
# Create a bar chart to compare purchase amounts by customer gender
sns.barplot(x='Gender', y='Purchase Amount (USD)', data=df)
plt.xlabel('Customer Gender')
plt.ylabel('Average Purchase Amount (USD)')
plt.title('Customer Gender vs. Purchase Amount')
plt.show()
from scipy.stats import ttest_ind

# Separate into Male and Female groups
male_data = df[df['Gender'] == 'Male']
female_data = df[df['Gender'] == 'Female']

# Perform a t-test to compare purchase amounts between Male and Female customers
t_stat, p_value = ttest_ind(male_data['Purchase Amount (USD)'], female_data['Purchase Amount (USD)'])

print(f"T-Statistic: {t_stat:.4f}")
print(f"P-Value: {p_value:.4f}")

# Interpret the results based on the p-value
if p_value < 0.05:
    print("There is a significant difference in Purchase Amount between Male and Female customers.")
else:
    print("There is no significant difference in Purchase Amount between Male and Female customers.")
# t-statistic is approximately -0.8769.
#The p-value is 0.3806, which is greater than the typical significance level of 0.05.
#Therefore, there is no significant difference in purchase amounts between Male and Female customers.

# %%
# Set style and context for better readability
sns.set(style="whitegrid")
sns.set_context("paper")

# Create a pairplot for multivariate analysis
g = sns.pairplot(df, vars=['Age', 'Purchase Amount (USD)'], height=3)
g.fig.suptitle("Pairplot of Age and Purchase Amount", y=1.02)  # Add a title
plt.show()

# %%
# Select numeric columns for correlation analysis
numeric_cols = df.select_dtypes(include='number')

# Create a correlation heatmap
correlation_matrix = numeric_cols.corr()
plt.figure(figsize=(10, 8))  # Adjust the figure size if needed
sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm', linewidths=0.5)
plt.title('Correlation Heatmap')
plt.show()

# %%
# Create a boxplot to compare purchase amounts by Size
sns.boxplot(x='Size', y='Purchase Amount (USD)', data=df)
plt.xlabel('Size')
plt.ylabel('Purchase Amount (USD)')
plt.title('Purchase Amount by Size ')
plt.show()

# %%
# Create a histogram to visualize the distribution of purchase amounts
plt.hist(df['Purchase Amount (USD)'], bins=20, color='skyblue', edgecolor='black')
plt.xlabel('Purchase Amount (USD)')
plt.ylabel('Frequency')
plt.title('Purchase Amount Distribution')
plt.show()

# %%
# Create a histogram to visualize the distribution of customer ages
plt.hist(df['Age'], bins=20, color='salmon', edgecolor='black')
plt.xlabel('Age')
plt.ylabel('Frequency')
plt.title('Customer Age Distribution')
plt.show()

# %%
# Create a violin plot to show the distribution of purchase amounts by location
plt.figure(figsize=(10, 6))  # Adjust the figure size
ax = sns.violinplot(x='Category', y='Purchase Amount (USD)', data=df)
ax.set_xlabel('Category', fontsize=12)  # Set x-axis label and font size
ax.set_ylabel('Purchase Amount (USD)', fontsize=12)  # Set y-axis label and font size
ax.set_title('Category-wise Purchase Amount Distribution', fontsize=14)  # Set title and font size
ax.set_xticklabels(ax.get_xticklabels(), fontsize=10, rotation=90)  # Adjust x-axis label font size and rotation

plt.show()

# %%
# Create a bar chart to show gender and Shipping Type counts
sns.countplot(x='Gender', hue='Shipping Type', data=df)
plt.xlabel('Gender')
plt.ylabel('Count')
plt.title('Gender and Shipping Type Counts')
plt.show()

# %%
# Create a bar chart to show the average purchase amount by color
plt.figure(figsize=(10, 6))
sns.barplot(x='Color', y='Purchase Amount (USD)', data=df)
plt.xlabel('Color')
plt.ylabel('Average Purchase Amount (USD)')
plt.title('Average Purchase Amount by color')
plt.xticks(rotation=45)
plt.show()

# %%
# Create a bar chart to show the average purchase amount byPayment Method
plt.figure(figsize=(10, 6))
sns.barplot(x='Payment Method', y='Purchase Amount (USD)', data=df)
plt.xlabel('Payment Method')
plt.ylabel('Average Purchase Amount (USD)')
plt.title('Average Purchase Amount by Payment Method')
plt.xticks(rotation=45)
plt.show()

# %%
# Create a bar chart to compare purchase amounts by season
plt.figure(figsize=(10, 6))  # Adjust the figure size
sns.barplot(x='Season', y='Purchase Amount (USD)', data=df, order=['Spring', 'Summer', 'Fall', 'Winter'])
plt.xlabel('Season', fontsize=12)
plt.ylabel('Average Purchase Amount (USD)', fontsize=12)
plt.title('Purchase Amount (USD) by Season', fontsize=14)

plt.show()

# EDA CODE BELOW ALSO USED IN SMART QUESTION.
# REMOVING EDA FROM HERE TO PREVENT REDUNDANCY.
# # %%
# # Create a scatter plot
# plt.figure(figsize=(8, 6))  # Adjust the figure size
# sns.scatterplot(x='Review Rating', y='Purchase Amount (USD)', data=df)
# plt.xlabel('Review Rating', fontsize=12)
# plt.ylabel('Purchase Amount (USD)', fontsize=12)
# plt.title('Purchase Amount (USD) vs. Review Rating', fontsize=14)

# plt.show()

    # %%
# Create a bar graph
plt.figure(figsize=(10, 6))  # Adjust the figure size
sns.barplot(x='Review Rating', y='Purchase Amount (USD)', data=df)
plt.xlabel('Review Rating', fontsize=12)
plt.ylabel('Purchase Amount (USD)', fontsize=12)
plt.title('Purchase Amount (USD) vs Review Rating', fontsize=14)
plt.xticks(rotation=0)  # Adjust the rotation angle of x-axis labels if needed

plt.show()

# %%
### DONE BY TEAM 
# Create 5 equal-sized bins for 'Age' and 10 equal-sized bins for 'Purchase Amount (USD)'.
age_bins = pd.cut(df['Age'], 5)
purchase_amount_bins = pd.cut(df['Purchase Amount (USD)'], 10)

# Compute the midpoints of the intervals for plotting
df['Age Group Midpoint'] = age_bins.apply(lambda x: x.mid)
df['Purchase Amount Group Midpoint'] = purchase_amount_bins.apply(lambda x: x.mid)

# Create a FacetGrid for 'Gender', mapping each 'Gender' to a different color.
g = sns.FacetGrid(df, col='Gender', hue='Gender', col_wrap=3, height=5)

# Map a scatter plot to each subset of the data.
g.map(plt.scatter, 'Age', 'Purchase Amount (USD)', alpha=0.6)

# Add a legend and title and adjust the layout.
g.add_legend()
g.fig.suptitle('Age vs Purchase Amount vs Gender', y=1.03)
g.fig.tight_layout()

# Rotate the x-axis labels for better readability.
for axes in g.axes.flat:
    _ = plt.setp(axes.get_xticklabels(), rotation=45)

# Show the plot.
plt.show()

# %%
import pandas as pd
from sklearn.preprocessing import LabelEncoder

# Assuming 'data' is the DataFrame containing your dataset
# Dropping non-numeric columns like 'Customer ID' and 'Promo Code Used' for correlation analysis
numeric_data = df.drop(columns=['Customer ID', 'Promo Code Used'])

# Encoding categorical variables
le = LabelEncoder()
for column in numeric_data.columns:
    if numeric_data[column].dtype == 'object':
        numeric_data[column] = le.fit_transform(numeric_data[column])

# Calculate the correlation matrix
correlation_matrix = numeric_data.corr()

# Print correlation matrix
print(correlation_matrix)

# %%
plt.figure(figsize =(20,6))
counts = df["Gender"].value_counts()
explode = (0,0.05)

counts.plot(kind = 'pie' ,fontsize = 12, colors = ["red","blue"], explode = explode, autopct = '%.1f%%')
plt.title('Males Vs Females')
plt.xlabel('Gender', weight = "bold", color = "#2F0F5D", fontsize = 14, labelpad = 20)
plt.ylabel('Counts', weight = "bold", color = "#2F0F5D", fontsize = 14, labelpad = 20)
plt.legend(labels = counts.index, loc = "best")

plt.show()

# %%
plt.figure(figsize=(20,6))

ax = df["Category"].value_counts().plot(kind= 'bar' , color = '#21cf4a' ,rot = 0)

for p in ax.patches:
     ax.annotate(int(p.get_height()), (p.get_x() + 0.25, p.get_height() + 1), ha = 'center', va = 'bottom', color = 'black')
     ax.tick_params(axis = 'both', labelsize = 15)

    
plt.xlabel('Item Gategory', weight = "bold", color = "#D71313", fontsize = 14, labelpad = 20)
plt.ylabel('Counts', weight = "bold", color = "#D71313", fontsize = 14, labelpad = 20)

# %%
# Define mappings for 'Category' and 'Location'
category_mapping = {'Clothing': 1, 'Accessories': 2, 'Footwear': 3, 'Outerwear': 4,"Blouse":5}
states = [
    "Kentucky", "Maine", "Massachusetts", "Rhode Island", "Oregon", "Wyoming", "Montana", "Louisiana",
    "West Virginia", "Missouri", "Arkansas", "Hawaii", "Delaware", "New Hampshire", "New York", "Alabama",
    "Mississippi", "North Carolina", "California", "Oklahoma", "Florida", "Texas", "Nevada", "Kansas",
    "Colorado", "North Dakota", "Illinois", "Indiana", "Arizona", "Alaska", "Tennessee", "Ohio", "New Jersey",
    "Maryland", "Vermont", "New Mexico", "South Carolina", "Idaho", "Pennsylvania", "Connecticut", "Utah",
    "Virginia", "Georgia", "Nebraska", "Iowa", "South Dakota", "Minnesota", "Washington", "Wisconsin",
    "Michigan"
]

state_numbers = {state: index + 1 for index, state in enumerate(states)}

Genger_mapping={"Male":1,"Female":0}
category_mapping ={"Clothing":1,"Footwear":2,"Outerwear":3,"Accessories":4}

df['Gender'] = df['Gender'].map(Genger_mapping)
df['Location']=df["Location"].map(state_numbers)
df["Category"]=df["Category"].map(category_mapping)

#%%
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import mean_squared_error, r2_score

# Load your dataset (replace 'data.csv' with your dataset file)
data = df

# Encode categorical variables
label_encoder = LabelEncoder()
data['encoded_category'] = label_encoder.fit_transform(data['Category'])
data['encoded_location'] = label_encoder.fit_transform(data['Location'])

# Define independent and dependent variables
X = data[['Age', 'Gender', 'Review Rating',"Location","Category","Gender"]]
y = data['Purchase Amount (USD)']

# Split data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Build and train the linear regression model
model = LinearRegression()
model.fit(X_train, y_train)

# Make predictions
y_pred = model.predict(X_test)

# Evaluate the model
mse = mean_squared_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)

print(f"Mean Squared Error: {mse}")
print(f"R-squared: {r2}")

# Analyze coefficients
coefficients = pd.DataFrame({'Variable': X.columns, 'Coefficient': model.coef_})
print(coefficients)

# %%
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
import pandas as pd

X = df['Location'].values.reshape(-1, 1)  # Predictor variable
y = df['Purchase Amount (USD)'].values  # Response variable

# create plot
plt.figure(figsize=(10, 6))

# scatterplot
plt.scatter(X, y, color='blue', label='Actual Data')

# create and bulid linearRegression
model = LinearRegression()
model.fit(X, y)
slope = model.coef_[0]
intercept = model.intercept_
# using model to predict
y_pred = model.predict(X)
plt.plot(X, y_pred, color='red', label='Regression Line')

plt.xlabel('Location')
plt.ylabel('Purchase Amount (USD)')
plt.title('Linear Regression: Location vs. Purchase Amount')
plt.legend()

plt.show()

print("Slope:", slope)
print("Intercept:", intercept)

#The red line in the graph indicates the line of best fit to the data. It indicates the general trend of the purchase amount as the customer's age changes.
#The slope of the regression line is approximately -0.0162, which means that for each additional year of age, the purchase amount decreases by approximately $0.016, with all other factors remaining constant.
#The intercept is approximately $60.48. Theoretically, this suggests that a customer who is zero years old would correlate to a purchase of approximately $60.48, it's not meaningful because the baby can't buy a car. 
#we can see the slope is very small, indicating a very weak negative correlation between age and purchase amount.

#Do correlation Test 
from scipy.stats import pearsonr

# Calculate Pearson's correlation coefficient and p-value
correlation, p_value = pearsonr(df['Location'], df['Purchase Amount (USD)'])

print(f"Pearson's Correlation: {correlation:.4f}")
print(f"P-Value: {p_value:.4f}")

# Interpret the results based on the p-value and correlation coefficient
if p_value < 0.05:
    print("There is a significant correlation between Location and Purchase Amount.")
else:
    print("There is no significant correlation between LOcation and Purchase Amount.")

#The correlation coefficient is approximately -0.0104, so indicating a very weak
#negative correlation between a customer's age and the purchase amount. And the p-value is 0.5152, which is greater than the typical significance level of 0.05.
#Therefore, you conclude that there is no significant correlation between  customer's age and the purchase amount. 

# %%
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
import pandas as pd

X = df['Category'].values.reshape(-1, 1)  # Predictor variable
y = df['Purchase Amount (USD)'].values  # Response variable

# create plot
plt.figure(figsize=(10, 6))

# scatterplot
plt.scatter(X, y, color='lightblue', label='Actual Data')

# create and bulid linearRegression
model = LinearRegression()
model.fit(X, y)
slope = model.coef_[0]
intercept = model.intercept_
# using model to predict
y_pred = model.predict(X)
plt.plot(X, y_pred, color='red', label='Regression Line')

plt.xlabel('Category')
plt.ylabel('Purchase Amount (USD)')
plt.title('Linear Regression:  Category vs. Purchase Amount')
plt.legend()

plt.show()

print("Slope:", slope)
print("Intercept:", intercept)

#The red line in the graph indicates the line of best fit to the data. It indicates the general trend of the purchase amount as the customer's age changes.
#The slope of the regression line is approximately -0.0162, which means that for each additional year of age, the purchase amount decreases by approximately $0.016, with all other factors remaining constant.
#The intercept is approximately $60.48. Theoretically, this suggests that a customer who is zero years old would correlate to a purchase of approximately $60.48, it's not meaningful because the baby can't buy a car. 
#we can see the slope is very small, indicating a very weak negative correlation between age and purchase amount.

#Do correlation Test 
from scipy.stats import pearsonr

# Calculate Pearson's correlation coefficient and p-value
correlation, p_value = pearsonr(df['Category'], df['Purchase Amount (USD)'])

print(f"Pearson's Correlation: {correlation:.4f}")
print(f"P-Value: {p_value:.4f}")

# Interpret the results based on the p-value and correlation coefficient
if p_value < 0.05:
    print("There is a significant correlation between  Category and Purchase Amount.")
else:
    print("There is no significant correlation between  Category and Purchase Amount.")

#The correlation coefficient is approximately -0.0104, so indicating a very weak
#negative correlation between a customer's age and the purchase amount. And the p-value is 0.5152, which is greater than the typical significance level of 0.05.
#Therefore, you conclude that there is no significant correlation between  customer's age and the purchase amount. 

# %%
#Is there a correlation between the purchase amount and the customer's
#re-purchase behavior (e.g. number of previous purchases) and the product category?
import seaborn as sns
import matplotlib.pyplot as plt

# Correlation analysis between 'Purchase Amount (USD)' and 'Previous Purchases'
##heatmap
correlation = df[['Purchase Amount (USD)', 'Previous Purchases']].corr()

# Visualization of this correlation
plt.figure(figsize=(10, 6))
sns.heatmap(correlation, annot=True, cmap='coolwarm', fmt=".2f")
plt.title("Correlation between Purchase Amount and Previous Purchases")
plt.show()

#The correlation coefficient of 0.01 between 'Purchase Amount (USD)' and 
#'Previous Purchases' suggests no significant linear relationship between 
#these variables, indicating that the frequency of a customer's previous 
#purchases does not linearly correlate with their spending amount.
 
# %%
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.preprocessing import OneHotEncoder
from sklearn.compose import ColumnTransformer

# For simplicity, we will select a subset of potential predictor variables for the regression model
# We will include both numeric and categorical variables and will use one-hot encoding for the categorical variables
numeric_features = ['Age', 'Review Rating', 'Previous Purchases']
categorical_features = ['Gender', 'Category', 'Season', 'Subscription Status', 'Discount Applied', 'Promo Code Used']

# Prepare the features DataFrame
X = df[numeric_features + categorical_features]
# Prepare the target variable
y = df['Purchase Amount (USD)']

# One-hot encode the categorical variables
preprocessor = ColumnTransformer(
    transformers=[
        ('num', 'passthrough', numeric_features),
        ('cat', OneHotEncoder(), categorical_features)])
X_processed = preprocessor.fit_transform(X)

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X_processed, y, test_size=0.2, random_state=42)

# Create and fit the regression model
regressor = LinearRegression()
regressor.fit(X_train, y_train)

# Predict on the test set
y_pred = regressor.predict(X_test)

# Calculate the performance metrics
mse = mean_squared_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)

mse, r2

#The high Mean Squared Error (MSE) in the model indicates a significant deviation between the 
#predicted and actual purchase amounts, suggesting a poor model fit. Additionally, the R²
#value is negative, implies that the model fails to effectively predict purchase amounts and 
#is not better than a simplistic mean-based model. These outcomes suggest that the model's 
#features lack a strong linear relationship with the 'Purchase Amount (USD)', and that exploring
#non-linear models or incorporating additional factors might improve predictive accuracy.

# %%
##Non-Linear Models
#Random Forest
from sklearn.ensemble import RandomForestRegressor

# Create and fit the random forest regressor
random_forest_regressor = RandomForestRegressor(n_estimators=100, random_state=42)
random_forest_regressor.fit(X_train, y_train)

# Predict on the test set
y_pred_rf = random_forest_regressor.predict(X_test)

# Calculate the performance metrics for the random forest model
mse_rf = mean_squared_error(y_test, y_pred_rf)
r2_rf = r2_score(y_test, y_pred_rf)

mse_rf, r2_rf

# Plot the actual vs predicted values using the Random Forest model for visualization
plt.figure(figsize=(12, 6))

# Plotting only the first 100 instances 
plt.scatter(range(len(y_test[:100])), y_test[:100], color='blue', label='Actual')
plt.scatter(range(len(y_pred_rf[:100])), y_pred_rf[:100], color='red', label='Predicted', alpha=0.8)

plt.title('Actual vs Predicted Purchase Amounts (USD) - Random Forest Model')
plt.xlabel('Index of Instance')
plt.ylabel('Purchase Amount (USD)')
plt.legend()
plt.show()
# In this plot I only present 100 instances, The scatter plot shows a mismatch between the
# actual purchase amounts (blue points) and the model's predictions (red points), reflecting
# the poor model fit indicated by the negative R² value. This suggests the need for model
# adjustments or exploring different predictive approaches.

# %%
# Calculating IQR for 'Purchase Amount (USD)'
Q1 = df['Purchase Amount (USD)'].quantile(0.25)
Q3 = df['Purchase Amount (USD)'].quantile(0.75)
IQR = Q3 - Q1

# Determining the bounds for outliers
lower_bound = Q1 - 1.5 * IQR
upper_bound = Q3 + 1.5 * IQR

# Filtering out outliers
data_no_outliers = df[(df['Purchase Amount (USD)'] >= lower_bound) & (df['Purchase Amount (USD)'] <= upper_bound)]

# Calculating the correlation in the cleaned dataset
cleaned_correlation = data_no_outliers['Purchase Amount (USD)'].corr(data_no_outliers['Previous Purchases'])
cleaned_correlation, df.shape[0], data_no_outliers.shape[0]  

#Removing outliers from the 'Purchase Amount (USD)' column did not change the correlation
#coefficient with the number of previous purchases, remaining at approximately 0.0081.
#The dataset size also remained the same, indicating no outliers were removed. This implies 
#the weak correlation observed is not influenced by outliers and suggests that purchase amount 
#does not significantly affect repeat purchases in this dataset.

#%%
##Group data by category
# Grouping the data by 'Category' and calculating the correlation within each category
category_correlations = data_no_outliers.groupby('Category').apply(
    lambda group: group['Purchase Amount (USD)'].corr(group['Previous Purchases'])
)

category_correlations
#These results suggest that within each product category, the correlation between purchase 
#amount and the customer's tendency to repurchase is very weak, whether slightly positive
#or negative. This reinforces the earlier finding that purchase amount does not significantly
#influence repurchase behavior across the dataset as a whole. The inclusion of product categories
#does not seem to change this conclusion significantly

# %%
#Previous Purchases VS Frequency of Purchases
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import scipy.stats as stats


# Method 1: Descriptive statistical analysis

frequency_grouped = df.groupby('Frequency of Purchases')['Previous Purchases'].mean()
print(frequency_grouped)

# Method 2: Visual analysis

plt.figure(figsize=(10, 6))
sns.boxplot(x='Frequency of Purchases', y='Previous Purchases', data=df)
plt.title('Previous Purchases by Frequency of Purchases')
plt.show()


# Method 3: Correlation analysis
# correlation = df['Frequency of Purchases'].corr(df['Previous Purchases'])
# print('Correlation:', correlation)

# Method 4: Chi-square test
contingency_table = pd.crosstab(df['Frequency of Purchases'], df['Previous Purchases'])
chi2, p, dof, expected = stats.chi2_contingency(contingency_table)
print('Chi-squared Test P-value:', p)

##In summary, from the data provided, there is no significant statistical relationship between 
#"Previous Purchases" and "Frequency of Purchases". This could mean that how often customers buy
#does not directly determine their buying history.


# %%
##Frequency of Purchases VS  Season

import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import scipy.stats as stats

#Crosstab analysis

contingency_table = pd.crosstab(df['Frequency of Purchases'], df['Season'])

#Visual analysis

plt.figure(figsize=(12, 8))
sns.heatmap(contingency_table, annot=True, fmt="d", cmap="YlGnBu")
plt.title("Frequency of Purchases vs Season")
plt.ylabel('Frequency of Purchases')
plt.xlabel('Season')
plt.show()

# Chi-square test

chi2, p, dof, expected = stats.chi2_contingency(contingency_table)
print('Chi-squared Test P-value:', p)

#try to find some relationship with purchase amount and other variables.
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import scipy.stats as stats
from sklearn.linear_model import LinearRegression

# # Correlation analysis

numeric_columns = df.select_dtypes(include=['int64', 'float64']).columns
correlations = df[numeric_columns].corrwith(df['Purchase Amount (USD)'])
print(correlations)

#Visualization
sns.scatterplot(x='Age', y='Purchase Amount (USD)', data=df)
plt.show()

# Multiple regression analysis
X = df[['Age', 'Previous Purchases', 'Review Rating']]  
y = df['Purchase Amount (USD)']
model = LinearRegression().fit(X, y)
print('Regression Coefficients:', model.coef_)

# ANOVA or T-test
# Select a categorical variable as an example
groups = df.groupby('Gender')['Purchase Amount (USD)'].apply(list)
f_val, p_val = stats.f_oneway(*groups)
print('ANOVA P-value:', p_val)

#Customer ID: The correlation coefficient is 0.011048, indicating a very weak correlation with the purchase amount.
#Age: The correlation coefficient is -0.010424, indicating little correlation with the purchase amount.
#Review Rating: The correlation coefficient is 0.030776, indicating a very weak positive correlation with the purchase amount.
#Previous Purchases: The correlation coefficient was 0.008063, indicating a very weak correlation with the purchase amount.


# %%
# Relationship between categorical variables and purchase amount
# try to make group
category_stats = df.groupby('Category')['Purchase Amount (USD)'].agg(['mean', 'median', 'count', 'std'])

# Visualize the distribution of purchase amounts for different product categories
sns.boxplot(x='Category', y='Purchase Amount (USD)', data=df)
plt.title('Purchase Amount by Product Category')
plt.show()


# non-linear relationship between age and purchase amount
from sklearn.preprocessing import PolynomialFeatures

age = df['Age'].values.reshape(-1, 1)

poly = PolynomialFeatures(degree=2, include_bias=False)
age_poly = poly.fit_transform(age)

model_poly = LinearRegression().fit(age_poly, df['Purchase Amount (USD)'])

print('Coefficients for Age and Age^2:', model_poly.coef_)

import numpy as np

df['Log_Purchase_Amount'] = np.log(df['Purchase Amount (USD)'] + 1)


# Try random forest regressors to capture more complex relationships
from sklearn.ensemble import RandomForestRegressor

rf = RandomForestRegressor(n_estimators=100, random_state=42)
rf.fit(X, df['Log_Purchase_Amount'])

# %%
#the relationship between Previous Purchases and Location

import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from scipy.stats import f_oneway

# Grouped Statistics
grouped_location = df.groupby('Location')['Previous Purchases'].agg(['mean', 'median', 'std', 'count'])

# Visualization
plt.figure(figsize=(14, 7))
sns.boxplot(x='Location', y='Previous Purchases', data=df)
plt.xticks(rotation=90)  # Rotate the x labels if there are many locations
plt.show()

# Statistical Tests
import pandas as pd
from scipy import stats

# Assuming 'df' is your DataFrame and it's already loaded with your data

# Group the 'Previous Purchases' data by 'Location'
grouped_data = df.groupby('Location')['Previous Purchases'].apply(list)

# Convert the GroupBy object to a list of arrays, one for each group
list_of_groups = [group for group in grouped_data]

# Perform ANOVA
anova_result = stats.f_oneway(*list_of_groups)

print(f"F-statistic: {anova_result.statistic}, p-value: {anova_result.pvalue}")

#This means that the location does not seem to have a statistically significant
#effect on the number of previous purchases made by customers



#%%
df1 = df.copy(deep=True)
# Calculate total sales for each item in each season
grouped_df1 = df1.groupby(['Season', 'Item Purchased']).agg({'Purchase Amount (USD)': 'sum'}).rename(columns={'Purchase Amount (USD)': 'Total Sales'}).reset_index()

# Find top 5 products in each season based on total sales
top_products_per_season = grouped_df1.groupby('Season').apply(lambda x: x.nlargest(5, 'Total Sales')).reset_index(drop=True)

# Separate the data by season
spring_data = top_products_per_season[top_products_per_season['Season'] == 'Spring']
summer_data = top_products_per_season[top_products_per_season['Season'] == 'Summer']
fall_data = top_products_per_season[top_products_per_season['Season'] == 'Fall']
winter_data = top_products_per_season[top_products_per_season['Season'] == 'Winter']

# Create a bar plot for each season
fig, axes = plt.subplots(nrows=2, ncols=2, figsize=(15, 10))
fig.suptitle('Top 5 Best-Selling Products per Season')

sns.barplot(x='Item Purchased', y='Total Sales', data=spring_data, ax=axes[0, 0]).set_title('Spring')
sns.barplot(x='Item Purchased', y='Total Sales', data=summer_data, ax=axes[0, 1]).set_title('Summer')
sns.barplot(x='Item Purchased', y='Total Sales', data=fall_data, ax=axes[1, 0]).set_title('Fall')
sns.barplot(x='Item Purchased', y='Total Sales', data=winter_data, ax=axes[1, 1]).set_title('Winter')

# Adjust layout for readability
plt.tight_layout(rect=[0, 0.03, 1, 0.95])
plt.show()

#%%
## SMART QUESTION: What is the relationship between review ratings and repeat purchases? 
# Using this analysis, we reveal if higher-rated products or services lead to more repeat purchases, indicating customer satisfaction.

plt.figure(figsize=(8, 6))  # Adjust the figure size
sns.scatterplot(x='Review Rating', y='Purchase Amount (USD)', data=df)
plt.xlabel('Review Rating', fontsize=12)
plt.ylabel('Purchase Amount (USD)', fontsize=12)
plt.title('Purchase Amount (USD) vs. Review Rating', fontsize=14)
plt.show()

plt.figure(figsize = (15, 6))
counts = df["Frequency of Purchases"].value_counts()
colors = sns.color_palette("tab10")
plt.title("Frequency of Purchases Distribution", weight="bold", fontsize=16)
counts.plot(kind='pie', fontsize=14, colors=colors, autopct='%1.1f%%', startangle=140)
plt.axis('equal')
plt.legend(labels=counts.index, loc="best", fontsize=12)
plt.show()

# Adjusting the bins for Previous Purchases so that 50-54 is the highest bin
df['Previous Purchases Grouped'] = pd.cut(df['Previous Purchases'], bins=[-1, 4, 9, 14, 19, 24, 29, 34, 39, 44, 49, 54], labels=['0-4', '5-9', '10-14', '15-19', '20-24', '25-29', '30-34', '35-39', '40-44', '45-49', '50-54'])

# Plot for Review Rating
plt.figure(figsize=(10, 6))
sns.countplot(x='Review Rating',hue='Review Rating', data=df, palette='viridis', legend=False)
plt.title('Distribution of Review Ratings')
plt.xlabel('Review Rating')
plt.ylabel('Count')
plt.show()

# Plot for Previous Purchases Grouped
plt.figure(figsize=(10, 6))
sns.countplot(x='Previous Purchases Grouped', hue='Previous Purchases Grouped', data=df, palette='Set2')
plt.title('Previous Purchases by Products')
plt.xlabel('Previous Purchases')
plt.ylabel('Count')
plt.show()


# Relationship between Review Ratings and Previous Purchases
plt.figure(figsize=(10, 6))
sns.boxplot(x='Review Rating', y='Previous Purchases', data=df)
plt.title('Review Ratings vs. Previous Purchases')
plt.show()

# %%
correlation = df[['Review Rating', 'Previous Purchases']].corr()
print(correlation, "\n")
print("Since the correlation coefficient (0.004229) between 'Review Rating' and 'Previous Purchases' is extremely close to 0, it suggests that there is virtually no linear relationship between these two variables.")
print("This means that changes in review ratings do not predictably affect the number of previous purchases, and vice versa, at least not in a linear manner.")

# %%
# Preparing the data
X = df[['Review Rating']]  # Predictor
y = df['Previous Purchases']  # Target

# Splitting the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Building the Linear Regression Model
model = LinearRegression()
model.fit(X_train, y_train)

# Predicting and Evaluating the Model
y_pred = model.predict(X_test)
linear_mse = mean_squared_error(y_test, y_pred)
linear_r2 = r2_score(y_test, y_pred)
print("Linear Regression Mean Squared Error:", linear_mse)
print("Linear Regression R-squared:", linear_r2)
# %%
from sklearn.tree import DecisionTreeRegressor

# Building the Decision Tree Regressor
tree_model = DecisionTreeRegressor(random_state=42)
tree_model.fit(X_train, y_train)

# Predicting and Evaluating the Model
tree_predictions = tree_model.predict(X_test)
tree_mse = mean_squared_error(y_test, tree_predictions)
tree_r2 = r2_score(y_test, tree_predictions)
print("Decision Tree Mean Squared Error:", tree_mse)
print("Decision Tree R-squared:", tree_r2)

# Feature Importance for Decision Tree
print("Feature Importance (Decision Tree):", tree_model.feature_importances_)

# %%
# Visual Analysis: Plotting Actual vs Predicted Values for Linear Regression
plt.scatter(X_test, y_test, color='blue', label='Actual')
plt.scatter(X_test, y_pred, color='red', alpha=0.5, label='Predicted')
plt.title('Linear Regression: Actual vs Predicted')
plt.xlabel('Review Rating')
plt.ylabel('Previous Purchases')
plt.legend()
plt.show()

# %%
# Conclusion
print("\nConclusion:")
print("The analysis with both models suggests a weak relationship between 'Review Rating' and 'Previous Purchases', as indicated by the high MSEs and low R-squared values.")
# %%
