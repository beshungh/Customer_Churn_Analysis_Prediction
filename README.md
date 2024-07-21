# Customer Churn Analysis and Prediction

### Project Overview
 The aim of this project is to predict customer churn for a bank using data analysis and machine learning techniques. This demonstrates the entire process, including data 
 preprocessing, exploratory data analysis, feature engineering, model building, evaluation, and recommendations.

## *Table of Contents*

- *[Introduction](#introduction)*
- *[Project Timeline](#project-timeline)*
- *[Data Analysis](#data-analysis)*
- *[Data Preprocessing](#tools-used)*
- *[Feature Engineering](#importing-csv-files-into-postgresql-using-python-script)*
- *[Model Building](#setbacks-of-the-python-script)*
- *[Findings And Recommendations](#entity-relationship-diagram)*
- *[Conclusion](#creating-database-tables)*

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

  
