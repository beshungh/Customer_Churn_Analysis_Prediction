# Customer Churn Analysis and Prediction

### Project Overview
 The aim of this project is to predict customer churn for a bank using data analysis and machine learning techniques. This demonstrates the entire process, including data 
 preprocessing, exploratory data analysis, feature engineering, model building, evaluation, and recommendations.

## *Table of Contents*

- *[Introduction](#introduction)*
- *[Project Timeline](#project-timeline)*
- *[Data Analysis](#data-analysis)*
- *[Data Preprocessing](#data-preprocessing)*
- *[Feature Engineering](#feature-engineering)*
- *[Model Building](#model-building)*
- *[Findings and Recommendations](#findings-and-recommendations)*
- *[Conclusion](#conclusion)*

### Introduction
Customer churn refers to the loss of clients or customers. Predicting customer churn helps businesses retain customers by identifying the signs early and taking action to prevent it. In this project, we use a dataset of bank customers to identify and visualize the key factors that contribute to customer churn and build a machine learning model that predicts whether a customer will churn.

### Project Timeline
The project will be executed in the following stages:

1. **Data Analysis**: Exploring and understanding the dataset, identifying trends and patterns related to customer churn.
2. **Data Processing**: Collecting, transformating, and organizing the raw data into a meaningful and useful format.
3. **Feature Engineering**: Creating and selecting relevant features that will improve the model's predictive power.
4. **Model Building using ANN**: Develop an Artificial Neural Network (ANN) model to predict customer churn.

### Data Analysis
#### Plotting Pie Chart 

```python
# Defining labels for the pie chart representing the two categories: 'Exited (Churned)' and 'Retained'
pie_labels = 'Exited (Churned)', 'Retained'

# Counting the number of customers who have exited (churned) and those who have been retained
pie_sizes = [
    df.Exited[df['Exited'] == 1].count(),  # Number of customers who exited
    df.Exited[df['Exited'] == 0].count()   # Number of customers who were retained
]

# The explode parameter is used to slightly separate the 'Retained' slice from the pie chart for better visualization.
slice_explode = (0, 0.1)  # No explode for 'Exited', and slight explode (0.1) for 'Retained'

# Creating a subplot with specific figure size
figure1, axis1 = plt.subplots(figsize=(10, 8))

# Plotting the pie chart
axis1.pie(
    pie_sizes,  # Sizes of the pie slices
    explode=slice_explode,  # Explode setting
    labels=pie_labels,  # Labels for the slices
    autopct='%1.1f%%',  # Display the percentage value with one decimal place
    shadow=True,  # Add a shadow to the pie chart for better visualization
    startangle=90  # Start the pie chart from a 90-degree angle
)

# Ensuring the pie chart is drawn as a circle
axis1.axis('equal')

# Adding a title to the pie chart
plt.title("Proportion of Customer Churned and Retained", size=20)

# Displays the pie chart
plt.show()
```
![Pie chart representing Churned and Retained](https://github.com/beshungh/Bank_Customer_Churn_Prediction/assets/135900689/8beaea3b-0cf4-4377-ad75-0c3fd148eb84)

Retained(0 or Blue)  and Exited(0 or Orange)

This pie chart shows the number of customers who have churned and the number of customers who have retained. 24.4% has exited the bank and 77.6 had retained.

#### Plotting a 2x2 Grid of Subplots - Countplot
```python
# Creating a 2x2 grid of subplots with a specified figure size
figure, axes_array = plt.subplots(2, 2, figsize=(20, 12)) # Creates a 2x2 grid of subplots within the figure, setting the figure size to 20x12 inches.

# Plotting a count plot for the 'Geography' column with a hue based on the 'Exited' column
# This shows the distribution of customers across different geographies, split by whether they exited or not
sns.countplot(x='Geography', hue='Exited', data=df, ax=axes_array[0][0]) # Places the plot in the first subplot of the grid.

# Plotting a count plot for the 'Gender' column with a hue based on the 'Exited' column
# This shows the distribution of customers based on gender, split by whether they exited or not
sns.countplot(x='Gender', hue='Exited', data=df, ax=axes_array[0][1]) # Places the plot in the second subplot of the grid.

# Plotting a count plot for the 'HasCrCard' column with a hue based on the 'Exited' column
# This shows the distribution of customers based on whether they have a credit card, split by whether they exited or not
sns.countplot(x='HasCrCard', hue='Exited', data=df, ax=axes_array[1][0]) # Places the plot in the third subplot of the grid.

# Plotting a count plot for the 'IsActiveMember' column with a hue based on the 'Exited' column
# This shows the distribution of customers based on their active membership status, split by whether they exited or not
sns.countplot(x='IsActiveMember', hue='Exited', data=df, ax=axes_array[1][1]) # Places the plot in the fourth subplot of the grid.

# Displaying the plots
plt.show()
```
![2x2 Grid of Subplots with a specified Figure Size](https://github.com/beshungh/Bank_Customer_Churn_Prediction/assets/135900689/1a1cbc85-e3f2-4e7c-83b7-6f05b6579f07)

1. Geography-Based Analysis and Recommendations

###### Analysis:

- France had the largest customer base among the three countries with approximately 6,000 customers. Despite having around 700 churned customers, the churn rate stands at     11.67%, which, while the highest in absolute numbers, is the lowest percentage-wise, indicating strong customer retention relative to the total number of customers.

- Spain had a moderate number of customers, totaling 2,000, with 400 of them churning, resulting in a churn rate of 20%. This indicates that one in five customers is
  churning, suggesting significant room for improvement in customer retention strategies.

- Germany has the fewest customers, totaling 1,700, but the highest churn rate at 47.06%, with 800 customers churning. Nearly half of the customers in Germany are churning,   indicating a significant issue with customer retention that requires immediate attention.

###### Recommendations:

* For France, the bank should continue current customer retention strategies and explore opportunities for customer expansion while maintaining retention.The bank can 
  continue offering loyalty programs like discounted loan rates and reward points for long-term customers. Additionally, they can explore new market segments by launching 
  innovative products such as digital wallets and sustainable investment funds to attract younger demographics while still focusing on retaining their current customers 
  through personalized financial advisory services.

* For Spain, the bank should implement targeted retention programs to reduce the churn rate and conduct customer satisfaction surveys to identify and address pain 
  points.These includes offering flexible loan repayment options and digital banking features like easy-to-use mobile apps. Also, the bank can conduct extensive customer 
  satisfaction surveys to gather feedback on service quality, waiting times, and product offerings. Based on survey results, the bank can introduce improvements such as 
  extended branch hours and enhanced online customer support to address identified pain points.

* For Germany, the bank should conduct a thorough analysis to understand the high churn rate, develop and implement aggressive retention strategies, and improve customer 
  service and engagement.Theses can be done through a detailed analysis to uncover reasons such as lack of personalized banking solutions and inefficient customer service. 
  In response, they develop aggressive retention strategies like personalized banking plans tailored to small business needs, including lower transaction fees and dedicated 
  account managers. Additionally, the bank should invests in customer service training programs and advanced CRM systems to improve overall customer engagement and 
  satisfaction.

2. Gender-Based Analysis and Recommendations

###### Analysis:

 The bank has more male customers (approximately 4,500) than female customers (approximately 3,500). However, the churn rate among female customers is higher, with about 
 1,100 female customers churning compared to 900 male customers. This results in a churn rate of 31.43% for female customers versus 20% for male customers, suggesting that 
 female customers might be facing unique challenges or dissatisfaction that needs to be addressed.

###### Recommendations:
The bank should: 

* investigate the reasons for higher churn among female customers through surveys and feedback with this, the bank can send out detailed surveys and feedback forms to 
  female customers who have recently churned and those who are still active. Questions can focus on their experiences with the bank’s services, specific pain points, and 
  suggestions for improvement.

* Develop targeted initiatives to address the specific needs and preferences of female customers. This can be achieved by introducing savings accounts with higher interest 
  rates, low-fee credit cards, or loan products specifically designed to support female entrepreneurs. Offering financial literacy workshops or investment seminars targeted 
  at women can also help empower female customers and address any financial knowledge gaps.

* Enhance customer support and engagement tailored for female customers.Hosting events and workshops that connect female customers with industry experts and successful 
  female entrepreneurs can foster a sense of community and loyalty towards the bank.

* Create a dedicated customer support team trained to handle inquiries and issues specific to female customers. This team can offer personalized financial advice, 
  empathetic service, and quick resolution of issues, ensuring that female customers feel valued and supported.

3. Credit Card Usage-Based Analysis and Recommendations

###### Analysis:

 Out of the total customer base, 5,500 customers hold credit cards, and among these, 1,600 have churned, resulting in a churn rate of approximately 29.09%. This is higher 
 than the overall churn rates, indicating that credit card holders are more likely to churn.

###### Recommendations:

For Credit Card Holder Retention,the bank should:

* Analyze the specific issues faced by credit card holders leading to higher churn. Example, the bank should Send surveys to churned credit card customers asking about 
  their reasons for leaving. Key areas to investigate include interest rates, fees, customer service experiences, reward programs, and any specific incidents that led to 
  their decision to churn. Also, analyze feedback from current credit card customers to identify ongoing pain points and areas for improvement.

* Improve the benefits and services associated with credit card usage. The bank should intriduce and enhance cashback offers, travel rewards, and points-based systems that 
  provide tangible benefits for using the credit card.They should ensure that these reward programs are competitive and clearly communicated to customers.

* Use data analytics to identify patterns in credit card usage and tailor communication to individual customers. Sending personalized offers, such as higher rewards for 
  categories they frequently spend on or special anniversary offers.

4. Activity-Based Analysis and Recommendations

###### Analysis:

 Among the customers, 3,500 are considered inactive, with 1,200 of them churning, resulting in a churn rate of approximately 34.29%. In contrast, there are 4,300 active 
 customers with 700 churning, leading to a churn rate of approximately 16.28%. This indicates that active customers are less likely to churn compared to inactive customers.

###### Recommendations:

For Activity-Based Retention the bank should:

*  Implement re-engagement campaigns for inactive customers to reduce their high churn rate. This can be done by sending personalized emails, messages, or direct mail to 
   inactive customers highlighting new features, products, or services. These communications can include special offers such as limited-time discounts or bonuses for 
   reactivating their accounts. For instance, an email campaign might offer a $50 bonus for using the credit card within the next month.

*  Provide incentives and personalized offers to encourage inactive customers to become active. Incentives like cashback, reward points, or lower fees for a specified 
   period 
   to inactive customers who start using their accounts again. For instance, a promotion could offer 5% cashback on purchases made with the bank's credit card for the next 
   three months.

*  Send regular, personalized updates to active customers about new products, services, and exclusive offers. This could include newsletters, app notifications, or social 
   media updates tailored to their usage patterns and preferences.

#### Plotting a 2x2 Grid of Subplots - Boxplot
```python
# Creating a 3x2 grid of subplots with a specified figure size
figure, axes_array = plt.subplots(3, 2, figsize=(20, 12))

# Plotting a boxplot for the 'CreditScore' column with 'Exited' as the x-axis, and use 'Exited' for hue
# This shows the distribution of credit scores split by whether the customer exited or not
sns.boxplot(y='CreditScore', x='Exited', hue='Exited', data=df, ax=axes_array[0][0])
axes_array[0][0].set_title('Credit Score vs Exited')

# Plotting a boxplot for the 'Age' column with 'Exited' as the x-axis, and use 'Exited' for hue
# This shows the distribution of age split by whether the customer exited or not
sns.boxplot(y='Age', x='Exited', hue='Exited', data=df, ax=axes_array[0][1])
axes_array[0][1].set_title('Age vs Exited')

# Plotting a boxplot for the 'Tenure' column with 'Exited' as the x-axis, and use 'Exited' for hue
# This shows the distribution of tenure split by whether the customer exited or not
sns.boxplot(y='Tenure', x='Exited', hue='Exited', data=df, ax=axes_array[1][0])
axes_array[1][0].set_title('Tenure vs Exited')

# Plotting a boxplot for the 'Balance' column with 'Exited' as the x-axis, and use 'Exited' for hue
# This shows the distribution of balance split by whether the customer exited or not
sns.boxplot(y='Balance', x='Exited', hue='Exited', data=df, ax=axes_array[1][1])
axes_array[1][1].set_title('Balance vs Exited')

# Plotting a boxplot for the 'NumOfProducts' column with 'Exited' as the x-axis, and use 'Exited' for hue
# This shows the distribution of the number of products split by whether the customer exited or not
sns.boxplot(y='NumOfProducts', x='Exited', hue='Exited', data=df, ax=axes_array[2][0])
axes_array[2][0].set_title('Number of Products vs Exited')

# Plotting a boxplot for the 'EstimatedSalary' column with 'Exited' as the x-axis, and use 'Exited' for hue
# This shows the distribution of estimated salary split by whether the customer exited or not
sns.boxplot(y='EstimatedSalary', x='Exited', hue='Exited', data=df, ax=axes_array[2][1])
axes_array[2][1].set_title('Estimated Salary vs Exited')

# Adjusting layout to prevent overlap
plt.tight_layout()

# Displaying the plots
plt.show()
```
![Plotting a 2x2 Grid of Subplots - Boxplot](https://github.com/user-attachments/assets/8986f100-8ae5-468a-9688-2986d3f195d8)

Credit Score-Based Analysis and Recommendations

###### Analysis:
- The analysis indicates that there is not much difference in churn rates between customers with different credit scores. This suggests that credit score is not a 
  significant factor in determining whether a customer will churn or not.

###### Recommendations:

For Credit Score-Based Retention, the bank should:

* Focus retention efforts on other more impactful factors since credit score is not a significant factor in churn, 
* Continue monitoring credit scores to ensure they do not become a significant factor in the future.

Age-Based Analysis and Recommendations

###### Analysis:
- The age-based analysis shows that older customers, particularly those above 40 years of age, are more likely to churn. In contrast, customers aged between 30 and 40 are 
  less likely to leave the bank. This suggests that age is a significant factor in customer retention, with older customers exhibiting higher churn rates.

###### Recommendations:

For Age-Based Retention, the Bank should:

* Develop specific strategies to retain older customers, who are more likely to churn.
* Consider offering age-specific incentives, services, and support to better meet the needs of older customers.
* Enhance engagement and communication efforts with older customers to improve their satisfaction and loyalty.

Tenure-Based Analysis and Recommendations

###### Analysis:
- The tenure-based analysis reveals that customers who have been with the bank for 7 years and above (oldest customers) are more likely to churn. Similarly, new members who 
  have been with the bank for a year or two are also more likely to churn. In contrast, customers who have been with the bank for an average of 4 to 6 years are less likely 
  to leave. This suggests that both new and long-term customers are at higher risk of churn compared to mid-tenure customers.

###### Recommendations: 

For Tenure-Based Retention, the bank should:

* Develop strategies to better retain both new and long-term customers, who are at higher risk of churn.
* Implement onboarding programs and personalized support for new customers to enhance their early experience.
* Recognize and reward long-term customers to maintain their loyalty and reduce the risk of churn.
* Focus on maintaining the satisfaction and engagement of mid-tenure customers who show lower churn rates.

Balance-Based Analysis and Recommendation

###### Analysis:
- The balance-based analysis indicates that customers with lower balances (0 to 90,000) are less likely to churn. Conversely, customers with higher balances (50,000 to 
  140,000) are more likely to churn. This suggests that both ends of the balance spectrum—those with very low and very high balances—are at higher risk of churn, whereas 
  customers with moderate balances tend to remain with the bank.

###### Recommendations: 

The Bank should:

* Provide personalized financial planning services to help them improve their financial health.
* Provide exclusive benefits or perks for maintaining high balances, such as premium banking services or VIP treatment.
* Implement proactive communication to keep them informed about relevant products or services.
* Develop programs to help customers optimize their account balances, such as balance transfer promotions or consolidation options.

Number of Product-Based Analysis and Recommendations

###### Analysis:
The analysis indicates that there is not much difference in churn rates between customers with number of products.
This suggests that number of product is not a significant factor in determining whether a customer will churn or not. 

###### Recommendations: 

The Bank should:

* Explore innovative approaches to differentiate products and services in a competitive market landscape.

Estimated Salary-Based Analysis and Recommendations

###### Analysis:
This analysis indicates that there is not much difference in churn rates between customers with estimated salary.
This suggests that estimated salary is not a significant factor in determining whether a customer will churn or not. 

###### Recommendations: 

The Bank should:

* Focus on other demographic or behavioral factors for segmenting customers. Consider factors such as age, location, purchase history, and product usage patterns.

### Data Preprocessing
#### Dropping Unnecessary Columns
 Dropping these columns helps the model to focus on the features that are truly predictive of customer churn, improving both the performance and interpretability of the  
 machine learning models.
 - CustomerId: If included, the model might treat each customer ID as a unique and significant factor, which could confuse the model and lead to poor generalization.
 - RowNumber: Similarly, the row number is just the sequence in which data is stored and carries no meaningful information about the customer.
 - Surname: While surnames might suggest familial relations or regional information, in most cases, they do not provide actionable insights for churn prediction.
```python
# Dropping the 'CustomerId', 'RowNumber', and 'Surname' columns from the DataFrame 'df'
# axis='columns' specifies that we're dropping columns (not rows)
# inplace=True means the changes will be applied directly to the original DataFrame without returning a new DataFrame

df.drop(['CustomerId','RowNumber','Surname'],axis='columns',inplace=True)
```

#### Encoding Categorical Variables
 Label encoding is a common technique used to convert categorical data into a numerical format that can be more easily understood and processed by machine learning 
 algorithms. Many algorithms require numerical input and cannot directly handle categorical data.

 The reason Why I am performing label encoding is most machine learning models, such as logistic regression, support vector machines, and neural networks, require numerical 
 input. They cannot process categorical data directly because they perform mathematical operations on the input data, which requires numerical values.
 ```python
# Replacing the values in the 'Gender' column: 'Male' with 1 and 'Female' with 0
df['Gender'].replace({'Male': 1, 'Female': 0}, inplace=True)
```

#### One Hot Encoding method
 In one-hot encoding, each category value is converted into a new binary column (or feature) where 1 indicates the presence of the category and 0 indicates the absence. 
 This creates a binary matrix where each column corresponds to one category and each row corresponds to one observation which allows categorical data to be used more 
 effectively in machine learning models, where numerical inputs are preferred.
 ```python
# Creating a new DataFrame 'df1' with one-hot encoded columns for the 'Geography' column
"""
pd.get_dummies: This function from the Pandas library is used to perform one-hot encoding.
data=df: Specifies the DataFrame df as the data source.
columns=['Geography']: Specifies the column(s) to be one-hot encoded. In this case, it's the 'Geography' column.
"""
df1 = pd.get_dummies(data=df, columns=['Geography'])

# Displaying the first few rows of the new DataFrame 'df1'
df1.head()
```

#### Scaling Features
```python
# Listing of variables to scale
scale_var = ['Tenure','CreditScore','Age','Balance','NumOfProducts','EstimatedSalary']

# Importing the MinMaxScaler module from scikit-learn library
from sklearn.preprocessing import MinMaxScaler

# Creating an instance of MinMaxScaler
scaler = MinMaxScaler()

# Scaling the specified columns (scale_var) in the DataFrame (df1)
df1[scale_var] = scaler.fit_transform(df1[scale_var])

df1.head()
```
- After scaling, the numerical columns are transformed. For example, CreditScore, Age, Tenure, Balance, NumOfProducts, and EstimatedSalary now have values between 0 and 1, 
  where 0 represents the minimum value in the original column, and 1 represents the maximum value. The categorical columns remain unchanged.
  For instance, CreditScore originally ranged from 350 to 850. After scaling, the minimum value becomes 0 and the maximum becomes 1, with other values scaled accordingly 
  within that range.
  This transformation makes the numerical features comparable and removes the potential bias introduced by differences in the scale of the original features. This 
  preprocessing step is often performed to improve the performance and stability of machine learning models.

### Feature Engineering
Creating new features or modifying existing ones to improve the predictive power of the model
```python
# Making a new column BalanceSalaryRatio
"""
Upon analysis, It was observed that neither the balance nor the estimated salary alone had a significant impact on customer churn.To explore the combined effect, a ratio 
of balance to estimated salary was calculated for each customer.
A box plot of this ratio was created to assess its relationship with churn.
"""
# Calculating the balance to salary ratio and adding it as a new column 'BalanceSalaryRatio' in the DataFrame
df['BalanceSalaryRatio'] = df['Balance'] / df['EstimatedSalary']

# Creating a boxplot to visualize the distribution of balance to salary ratio for customers who churned and those who didn't
# The x-axis represents the 'Exited' status, and the hue differentiates between churned and retained customers
# Set the y-axis limit to -1 and 5 for better visualization
sns.boxplot(y='BalanceSalaryRatio', x='Exited', hue='Exited', data=df)
plt.ylim(-1, 5)  # Set the y-axis limit

![BalanceSalaryRatio](https://github.com/user-attachments/assets/ca4db7f2-96a7-42dc-9d34-ebfa4324319c)

###### Analysis:
The box plot analysis indicates that customers with a balance-to-salary ratio around 2 are more likely to churn.

###### Recommendations:

The company should:

* Develop personalized communication and engagement strategies for customers with a balance-to-salary ratio around 2.
* Provide tailored financial advice or offers to improve their satisfaction and reduce the likelihood of churn.
```

```python
# Calculating the tenure to age ratio and adding it as a new column 'TenureByAge' in the DataFrame
"""
df['TenureByAge']: Creates a new column in the DataFrame to store the calculated ratios.
df['Tenure']: Accesses the Tenure column in the DataFrame.
df['Age']: Accesses the Age column in the DataFrame.
df['Tenure'] / df['Age']: Performs element-wise division to calculate the tenure to age.
df['TenureByAge']: Creates a new column in the DataFrame to store the calculated ratios.
"""
df['TenureByAge'] = df['Tenure'] / df['Age'] # Performs element-wise division to calculate the tenure to age ratio.

# Creating a boxplot to visualize the distribution of tenure to age ratio for customers who churned and those who didn't
# The x-axis represents the 'Exited' status, and the hue differentiates between churned and retained customers
# Set the y-axis limit to -1 and 1 for better visualization
"""
y='TenureByAge': The TenureByAge column will be on the y-axis.
x='Exited': The Exited column will be on the x-axis.
hue='Exited': Colors the boxes based on the Exited status.
data=df: Specifies the DataFrame df as the data source
"""
sns.boxplot(y='TenureByAge', x='Exited', hue='Exited', data=df)
plt.ylim(-1, 1)  # Set the y-axis limit

# Display the plot
plt.show()
```

![TenureByAge](https://github.com/user-attachments/assets/c8b25162-5752-478a-b2ac-5b7b90f407c2)

###### Analysis:
The analysis shows no significant relationship between the TenureByAge ratio and churn. Most customers have a TenureByAge ratio between 0.00 and 0.25, indicating minimal 
variation.

###### Recommendations:
The company should:

* Implement onboarding programs that emphasize long-term benefits and loyalty programs for younger customers with low tenure.
* Offer rewards and recognition programs to reinforce their loyalty for older customers with long tenure.
* Provide financial planning services that cater to their stage in life, such as retirement planning.

### Model Building
#### Train-Test Split
Splitting the dataset into training and testing sets to evaluate the model's performance
```python
# Separating independent features (X) and dependent feature (y)
"""
X: Contains the independent features. It's created by dropping the 'Exited' column from the DataFrame df1.
y: Contains the dependent feature, which is the 'Exited' column from the DataFrame df1.
X = df1.drop('Exited', axis='columns'): This line creates a new DataFrame X by dropping the column labeled 'Exited' from the DataFrame df1. The parameter axis='columns' specifies that we want to drop a column (as opposed to dropping a row), and dropping by label ('columns') means we are specifying the name of the column to drop.
"""
X = df1.drop('Exited', axis='columns')  # Independent features (excluding the 'Exited' column)
y = df1['Exited']  # Dependent feature (the 'Exited' column)

# Importing the train_test_split function from scikit-learn
# sklearn.model_selection is used to split the dataset into training and testing sets.

from sklearn.model_selection import train_test_split

# Splitting the data into training and testing sets
# The test_size parameter specifies the proportion of the dataset to include in the test split (here, 20%)
# The random_state parameter sets the random seed for reproducibility
"""
X_train: Contains the independent features for training.
X_test: Contains the independent features for testing.
y_train: Contains the dependent feature for training.
y_test: Contains the dependent feature for testing.
train_test_split(X, y, test_size=0.2, random_state=5): This function splits the dataset into training and testing sets.
X: The independent features.
y: The dependent feature.
test_size=0.2: Specifies that 20% of the data will be used for testing, while 80% will be used for training.
random_state=5: Sets the random seed to 5 for reproducibility, ensuring that the same random split is obtained each time the code is run.
"""
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=5)
```
The ultimate goal of a machine learning model is to generalize well to unseen data. Splitting the dataset into training and testing sets helps ensure that the model is not 
biased towards the specific instances in the training set and can make accurate predictions on new data.

### Model Selection
#### The Sequential model
The Sequential model is a linear stack of layers in deep learning, particularly in the context of Keras, a high-level neural networks API written in Python and capable of 
running on top of TensorFlow.
##### The Sequential model is not appropriate when:
* The model has multiple inputs or multiple outputs.
* Any of the layers has multiple inputs or multiple outputs.
* The need to do layer sharing.
* Want non-linear topology, such as a residual connection or a multi-branch model.

### Model Training 
```python
# Importing TensorFlow and Keras
"""
TensorFlow is the open-source machine learning library developed by Google, and Keras is an API that runs on top of TensorFlow, 
providing a high-level interface for building and training neural networks.
"""
import tensorflow as tf
from tensorflow import keras

# Defining the model architecture
"""
The neural network model is defined using the Sequential API from Keras. This model consists of three layers:
Input layer: Dense layer with 12 neurons and input shape (14,). This layer uses ReLU (Rectified Linear Unit) activation function.
Hidden layer: Dense layer with 6 neurons and ReLU activation function.
Output layer: Dense layer with 1 neuron and Sigmoid activation function. Sigmoid is used for binary classification tasks as it squashes the output between 0 and 1, making 
it suitable for probability predictions.
"""
model = keras.Sequential([
    keras.layers.Input(shape=(14,)),  # Correctly specifying input shape
    keras.layers.Dense(12, activation='relu'),  # Input layer with 12 neurons, using ReLU activation
    keras.layers.Dense(6, activation='relu'),  # Hidden layer with 6 neurons, using ReLU activation
    keras.layers.Dense(1, activation='sigmoid')  # Output layer with 1 neuron, using Sigmoid activation for binary classification
])

# Compiling the model
"""
Before training, the model needs to be compiled. During compilation, i specified the optimizer, loss function, and metrics to be used during training.
optimizer='adam': Adam optimizer, a popular optimization algorithm.
loss='binary_crossentropy': Binary cross-entropy loss function, commonly used for binary classification tasks.
metrics=['accuracy']: Accuracy metric will be monitored during training.
"""
model.compile(optimizer='adam',  # Optimizer algorithm
              loss='binary_crossentropy',  # Loss function for binary classification
              metrics=['accuracy'])  # Evaluation metric to monitor during training

# Training the model
"""
The model is trained using the fit method. It takes the training data (X_train and y_train) and the number of epochs (how many times the model will see the entire training 
dataset).
"""
model.fit(X_train, y_train, epochs=100)  # Training the model with X_train and y_train for 100 epochs
```

### Model Evaluation
```python
# Evaluating the trained model on the test data to assess its performance.
# X_test: the input features of the test dataset
# y_test: the corresponding true labels of the test dataset
model.evaluate(X_test, y_test)
```

### Predictions 
```python
# making prediction using machine learning model
"""model: This is your trained machine learning model.
predict(): This method is used to make predictions based on the input data provided.
X_test: This is the test dataset that you are using to evaluate the model's performance.
yp: This variable stores the predictions made by the model.
"""
yp = model.predict(X_test)
yp
```

### Checking Accuracy
```python
# Importing the confusion_matrix and classification_report functions from sklearn.metrics
from sklearn.metrics import confusion_matrix, classification_report

# Printing the classification report to evaluate the performance of the classification model
# y_test: ground truth (correct) labels
# y_pred: predicted labels by the classification model
print(classification_report(y_test, y_pred))
```
```python
# Importing seaborn as sn for data visualization and TensorFlow for computing the confusion matrix
import seaborn as sn
import tensorflow as tf
import matplotlib.pyplot as plt  # Import matplotlib for plotting

# Compute the confusion matrix using TensorFlow
# labels=y_test: ground truth (correct) labels
# predictions=y_pred: predicted labels by the classification model
cm = tf.math.confusion_matrix(labels=y_test, predictions=y_pred)

# Creating a plot with a specified size
plt.figure(figsize=(10, 7))

# Using seaborn to create a heatmap for the confusion matrix
# cm: the confusion matrix to be visualized
# annot=True: write the data value in each cell
# fmt='d': format the annotations as integers
sn.heatmap(cm, annot=True, fmt='d')

# Label the x-axis as 'Predicted'
plt.xlabel('Predicted')

# Label the y-axis as 'Truth'
plt.ylabel('Truth')

# Display the plot
plt.show()
```
![confusion matrix](https://github.com/user-attachments/assets/1b764aa7-2ca7-4416-a108-4e293dbb36c1)

```python
# Importing accuracy_score from sklearn.metrics
from sklearn.metrics import accuracy_score
```
```python
# Printing the accuracy score
"""
Accuracy score is:This is a string that will be printed as is.

accuracy_score(y_test,y_pred): This part calculates the accuracy score. accuracy_score is a function from a machine learning library like scikit-learn. It takes two parameters:

y_test: This is the true labels or target values of the test dataset.

y_pred: This is the predicted labels or target values generated by a machine learning model for the test dataset.

*100: This multiplies the accuracy score by 100 to convert it into a percentage. The accuracy score is typically a fraction or decimal between 0 and 1, and multiplying by 100 converts it into a percentage.

The:.2f inside the curly braces specifies that the accuracy should be displayed as a floating-point number with two decimal places.

"%": This is another string, representing the percentage sign that will be printed after the accuracy score.
"""
accuracy = accuracy_score(y_test, y_pred) * 100
print(f"Accuracy score is: {accuracy:.2f}%")
```

### Findings and Recommendations



### Conclusion






