
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
ax = sns.violinplot(x='Location', y='Purchase Amount (USD)', data=df)
ax.set_xlabel('Location', fontsize=12)  # Set x-axis label and font size
ax.set_ylabel('Purchase Amount (USD)', fontsize=12)  # Set y-axis label and font size
ax.set_title('Location-wise Purchase Amount Distribution', fontsize=14)  # Set title and font size
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
