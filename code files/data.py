
#%%
import seaborn as sns
import matplotlib.pyplot as plt
import pandas as pd
import missingno as msno


#%%
df = pd.read_csv('shopping_trends.csv')
df
# %%
#summary of the data set
df.describe()
# %%
df.isnull().sum()
# %%
# Create a heatmap to visualize missing values
plt.figure(figsize=(10, 6))
sns.heatmap(df.isnull(), cbar=False, cmap='viridis')
plt.title('Missing Values in the Dataset')
plt.show()
#%%
plt.figure(figsize = (20, 6))
ax = df["Gender"].value_counts().plot(kind = 'bar', color = ["blue","pink"], rot = 0)
ax.set_xticklabels(('Male', 'Female'))
plt.title("male and female population",weight="bold")
for p in ax.patches:
    ax.annotate(int(p.get_height()), (p.get_x() + 0.25, p.get_height() + 1), ha = 'center', va = 'bottom', color = 'black')
    ax.tick_params(axis = 'both', labelsize = 15)
plt.xlabel('Employment Type', weight = "bold", color = "#D71313", fontsize = 14, labelpad = 20)
plt.ylabel('Number of Occurrences', weight = "bold", color = "#D71313", fontsize = 14, labelpad = 20);
#%%
fig, ax = plt.subplots(figsize = (20, 5))

ax.hist(df['Age'], bins = 25, edgecolor = 'black', alpha = 0.7, color = 'skyblue', density = True)
df['Age'].plot(kind = 'kde', color = 'red', ax = ax)

ax.set_xlabel('Age')
ax.set_ylabel('Count / Density')
ax.set_title('Age Distribution Histogram with Density Curve')
ax.legend(['Density Curve', 'Histogram'])
plt.show()
#%%
plt.figure(figsize = (20, 6))
ax = df["Category"].value_counts().plot(kind = 'bar', color = ["red","blue","pink","yellow"], rot = 0)
ax.set_xticklabels(('Clothing', 'Accessories', 'Footwear', 'Outerwear'))
plt.title("Category")
for p in ax.patches:
    ax.annotate(int(p.get_height()), (p.get_x() + 0.25, p.get_height() + 1), ha = 'center', va = 'bottom', color = 'black')
    ax.tick_params(axis = 'both', labelsize = 15)
plt.xlabel('Employment Type', weight = "bold", color = "#D71313", fontsize = 14, labelpad = 20)
plt.title("Category vs frequency")
plt.ylabel('Number of Occurrences', weight = "bold", color = "#D71313", fontsize = 14, labelpad = 20);
#%%

plt.figure(figsize = (16, 6))
df["Location"].value_counts()[:10].sort_values(ascending = False).plot(kind = 'bar', color = sns.color_palette('inferno'), edgecolor = 'black')
plt.xlabel('Location', weight = "bold", color = "#D71313", fontsize = 14, labelpad = 20)
plt.ylabel('\nNumber of Occurrences', weight = "bold", color = "#D71313", fontsize = 14, labelpad = 20);
plt.xticks(rotation = 0, ha = 'center', fontsize = 16)
plt.title("Location vs frequency")
plt.tight_layout()
plt.show()
#%%
plt.figure(figsize = (20, 6))
ax = df["Size"].value_counts().plot(kind = 'bar', color = ["red","blue","green","yellow"], rot = 0)
ax.set_xticklabels(('Medium', 'Large', 'Small', 'Extra Large'))
plt.title("Size vs frequency")
for p in ax.patches:
    ax.annotate(int(p.get_height()), (p.get_x() + 0.25, p.get_height() + 1), ha = 'center', va = 'bottom', color = 'black')
    ax.tick_params(axis = 'both', labelsize = 15)
plt.xlabel('Size', weight = "bold", color = "#D71313", fontsize = 14, labelpad = 20)
plt.ylabel('Number of Occurrences', weight = "bold", color = "#D71313", fontsize = 14, labelpad = 20);
#%%
plt.figure(figsize = (16, 6))
df["Color"].value_counts()[:10].sort_values(ascending = True).plot(kind = 'barh', color = sns.color_palette('tab20'), edgecolor = 'black')
plt.xlabel('Color', weight = "bold", color = "#D71313", fontsize = 14, labelpad = 20)
plt.ylabel('\nNumber of Occurrences', weight = "bold", color = "#D71313", fontsize = 14, labelpad = 20);
plt.xticks(rotation = 0, ha = 'center', fontsize = 16)
plt.title("Colour vs frequency")
plt.tight_layout()
plt.show()
#%%
plt.figure(figsize = (20, 6))

counts = df["Season"].value_counts()
explode = (0, 0, 0, 0)

counts.plot(kind = 'pie', fontsize = 12, colors = ["red","blue","green","yellow"], explode = explode, autopct = '%1.1f%%')
plt.xlabel('Size', weight = "bold", color = "#2F0F5D", fontsize = 14, labelpad = 20)
plt.title("Season vs Size")
plt.axis('equal')
plt.legend(labels = counts.index, loc = "best")
plt.show()
#%%
plt.figure(figsize = (20, 6))
ax = df["Subscription Status"].value_counts().plot(kind = 'bar', color =["red","blue"] , rot = 0)
ax.set_xticklabels(('No', 'Yes'))
plt.title("Subscription Status")
for p in ax.patches:
    ax.annotate(int(p.get_height()), (p.get_x() + 0.25, p.get_height() + 1), ha = 'center', va = 'bottom', color = 'black')
    ax.tick_params(axis = 'both', labelsize = 15)
plt.xlabel('Subscription Status', weight = "bold", color = "#D71313", fontsize = 14, labelpad = 20)
plt.ylabel('Number of Occurrences', weight = "bold", color = "#D71313", fontsize = 14, labelpad = 20);

#%%
plt.figure(figsize = (20, 6))

counts = df["Payment Method"].value_counts()
explode = (0, 0, 0, 0, 0.0, 0.06)
plt.title("Payment Method",weight="bold")
counts.plot(kind = 'pie', fontsize = 12, colors = ["red","blue","green","yellow","pink","purple"], autopct = '%1.1f%%')
plt.xlabel('Size', weight = "bold", color = "#2F0F5D", fontsize = 14, labelpad = 20)
plt.axis('equal')
plt.legend(labels = counts.index, loc = "best")
plt.show()
#%%
plt.figure(figsize = (20, 6))

counts = df["Frequency of Purchases"].value_counts()
explode = (0, 0, 0, 0, 0.0, 0, 0.06)
plt.title("Frequency of Purchases",weight="bold")
counts.plot(kind = 'pie', fontsize = 12, colors  = ["red","blue","green","yellow","pink","purple","orange"], autopct = '%1.1f%%')
plt.xlabel('Size', weight = "bold", color = "#2F0F5D", fontsize = 14, labelpad = 20)
plt.axis('equal')
plt.legend(labels = counts.index, loc = "best")
plt.show()
# %%


# Visualize the missing values using a matrix
msno.matrix(df)
plt.title('Missing Values Matrix')
plt.show()
#%%


# Count missing values per column
missing_counts = df.isnull().sum()
plt.figure(figsize=(12, 6))
sns.barplot(x=missing_counts.index, y=missing_counts.values)
plt.xlabel('Columns')
plt.ylabel('Number of Missing Values')
plt.title('Count of Missing Values per Column')
plt.xticks(rotation=90)
plt.show()

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



# Create a scatter plot
plt.figure(figsize=(8, 6))  # Adjust the figure size
sns.scatterplot(x='Review Rating', y='Purchase Amount (USD)', data=df)
plt.xlabel('Review Rating', fontsize=12)
plt.ylabel('Purchase Amount (USD)', fontsize=12)
plt.title('Purchase Amount (USD) vs. Review Rating', fontsize=14)

plt.show()

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
plt.ylabel('Counts', weight = "bold", color = "#D71313", fontsize = 14, labelpad = 20);
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
df.isnull

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



