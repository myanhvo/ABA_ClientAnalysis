<h1 align="center"> Enhancing Customer Experience in the Legal Services Industry: Leveraging ABA's Raw Data for Predictive Analysis


<h2 align="center">  Abstract:

 
The purpose of this study is to uncover trends and evaluate customer satisfaction with the American Bar Association’s pro bono legal services across the United States via an online platform. The platform allows state-based low-income status people to seek legal advice by posting legal questions and receiving assistance from volunteer lawyers. By analyzing customer demographics, question categories, sentiment analysis, and factors influencing frequently asked questions of this service, we employ advanced machine learning techniques, including elastic net, random forests, and neural networks, to build a robust classification model. This model helps predict the most popular question categories based on key predictor variables while assessing a few drawbacks that occurred. Our research highlights the critical role of effective communication and expertise in maximizing client satisfaction and offers valuable recommendations for enhancing law firm services based on our machine learning model.

  <hr>
  
 

### 1.  BACKGROUND AND INTRODUCTION
    

  

In this study, our objective is to analyze the provided data to identify patterns and trends in client-lawyer conversations that can inform the ABA's advice to state partners, facilitate the creation of resources tailored to identified patterns, and devise outreach strategies to effectively engage potential users and volunteers. We aim to enhance the connection between lawyers and clients, promoting effective communication by utilizing similar language, understanding the cultural, societal, and emotional context of clients' messages, and facilitating active listening and engagement.  

The data at hand provides actual conversations between clients and lawyers and basic demographic information about the clients. The exchanges have already been categorized into various categories and sub-categories, although further classification adjustments are made by platform administrators. Through applying advanced machine learning techniques, including elastic net, random forests, decision trees, and neural networks, a robust classification model capable of accurately will be developed to predict the most popular question categories based on key predictor variables.  

  

### 2.  EXPLORATORY DATA ANALYSIS
    

  

Our dataset consists of real conversations between lawyers and clients from the American Bar Association's (ABA) legal services. It includes demographic information and covers conversations from August 5, 2015, to January 24, 2022. The study focused on 65,536 conversations, analyzing variables like sex, age, state names, and income-related subjects. ABA's services experienced significant growth across the United States, with the majority of customers coming from the South and East regions, but less popular in the North and West regions. The customer segment primarily consists of Caucasian, Latinx, and African people aged 21 to 50. For example, work-related questions are most common in Florida and Illinois, while juvenile inquiries are frequent in Texas, Florida, etc. Overall, analyzing these categories alongside sentiment variables helps identify patterns among ABA's customers’ inquiries and concerns.

  

### 3.  METHODS
   
#### 3.1. Sentiment Analysis

  
![](https://lh3.googleusercontent.com/aF4htuYCVuJz_gUNL_-80nUQAp89EVB2fnaN4lJxIegEHG8At9QUJqbR6PXNe9r5dcEJtU5qhtduIlDXFdy6l75fpdVYp3yD0XgITlngoDhr7noWM0ylSLVPVs9CH6HDMBuNm8DpanHva0cX6NU-D6Q)  
![](https://lh3.googleusercontent.com/DP8E2-A8Ns9UDh_Elebf0odExbsGLCXYsoo-kkuBCNcmMe4sJ963-LHdlNjyDfDGIOWucrmJ-3OlJbt_CFRTWfqr7KyN-UV6K9UrkQJyvDb-qvvXNiViYatXgKpc6eWf8SOhOAKj28ylAvfQ2L5qgW8)![](https://lh5.googleusercontent.com/NTNYZbDMp1LWVtDacG_EoK_Cmzmcp1ziVoVXcOtFPRa_C8wZL5D6gut57L6yNsBfKbUDqcbjiitl8oeu4I0FK8Y5V6xBhAzg2enYMy8xIMC2IDU7T8KYKNYGSOxzfgYv3pc_2p1c4dkUN9GBALWlfAM)  


#### Figure 1: (From left to right) Sentiment scores displayed for 3 out of 11 categories: Family and Children, Individual Rights, and Work, Employment, and Unemployment.

Chats and phone calls history between ABA agents and clients were recorded. Based on the conversations, sentiment analysis was performed on eleven specific conversation categories within the ABA, revealing eight sentiment score groups. Trust consistently received the highest score, surpassing the second-highest sentiment by 50%. High anticipation was observed across all categories. Among the three prominent categories (Family and Children, Individual Rights, and Health and Disability), Family and Children showed a significantly low sadness sentiment, while Health and Disability displayed the highest sadness sentiment score. Surprisingly, the Individual Rights category exhibited elevated levels of fear and anger, suggesting concerns regarding the quality of services provided.

  

#### 3.2. Predictive Analysis

  

This section outlines the methodology used to predict popular questions based on customer demographics for ABA. Four machine learning models, including Elastic Nets, Random Forest, Neural Networks, and Decision Trees, were trained and tested. The goal was to identify influential factors in determining question categories to improve customer service.

The first model that is used in this report is elastic net – a combination of the strengths of L1 regularization (encouraging sparse solutions by setting some coefficients to zero) and L2 regularization (shrinking the coefficients towards zero). Random forest is chosen as our second model because there are many predictors with complex interactions and non-linearity relationships between the predictors and the response. Neural networks are the third chosen model, which uses a flexible algorithm that can model highly complex relationships between the predictors and the response. This model is very computationally expensive to train though it can help reduce overfitting. Lastly, decision trees, a flowchart-like structure that makes decisions based on input features, resulting in a hierarchical sequence of binary splits, were employed. Decision trees are advantageous for classification tasks as they are easily interpretable and robust to outliers and missing values.

| Machine learning algorithm | Hyper-parameters                            | Values used for optimization                     |
|---------------------------|---------------------------------------------|-------------------------------------------------|
| Elastic Net               | Alpha (balance between L1 and L2 penalties) | 0, 0.2, 0.4, 0.6, 0.8, 1                        |
|                           | Lambda (L2 regularization parameter)       | 0, 0.02, 0.04, 0.06, 0.08, 0.10                |
| Random Forest             | Max features                                | 1 to 11                                         |
|                           | Number of trees                             | 50                                              |
| Neural Networks           | Size (Hidden Units)                         | 2, 5, 6                                         |
|                           | Decay (Regularization)                      | 0.6, 0.8, 0.1                                   |
| Decision Tree             | Max Depth                                   | 10, 11, 12, 13, 14, 15                          |
|                           | Cp (Complexity of parameter)                | 0.00001 to 0.000015                             |
|                           | Min Split                                   | 1, 2, 3, 4, 5                                  |

#### Table 1: Hyper-parameter values considered for optimization.

  

#### 3.3 Optimization and Performance Evaluation
Optimizing an ML model consists of finding the best hyper-parameter values (Table 1), involving adjusting parameters depending on the specific algorithm employed. 80% of the data are allocated for training and the remaining 20% for testing. To evaluate the model's performance and assess its ability to generalize, a 5-fold cross-validation strategy was employed for all models except Random Forest, which used the "Out-of-Bag" Estimation method. Addressing the issue of imbalanced data was crucial in this study. Techniques such as oversampling and undersampling were utilized to achieve a more balanced representation of the classes, thereby improving the model's ability to learn from the minority classes.

Performance metrics are generated from confusion matrices to assess the performance of the models. These includes accuracy, misclassification rate, precision, recall, and F1 score. Accuracy represents overall correctness, misclassification rate measures the proportion of incorrect predictions, precision evaluates the model's ability to identify positive instances correctly, recall assesses the model's capability to identify all positive instances, and the F1 score provides a balanced measure of precision and recall. These metrics enable a comprehensive evaluation of the model's predictive capabilities and help in selecting the most effective approach for predicting popular questions based on customer demographics.

### 4.  RESULT
    

![](https://lh4.googleusercontent.com/ziVRfZSiGsFuRfzCUypxZgAAcUatr7nNke4TjfdTbdVJSHzE76QnUaIXGAkUFl6KEhjnkKZiFmajQ8m49TabgyamxTqTH-LwiHNe1ptQNVVE8J805K3fimJ1a6r7Nxi9pPnglw26xQahTYEIGUvwZyY)

#### Table 2: Performance metrics across four models

  

Before selecting the optimal model for prediction, several criteria should be considered. The objective is to maximize accuracy while minimizing false positives and false negatives. False positives (low precision) allocate unnecessary resources, while false negatives (low recall) result in missed important customer queries.

Among the models evaluated, the Random Forests model demonstrates superior performance across multiple evaluation criteria. It achieves the highest accuracy of 64.8%, indicating a greater overall number of correct predictions. Furthermore, Random Forests achieves the highest precision and the highest recall. The model also obtains the highest F1 score, which balances precision and recall. Considering these metrics collectively, the Random Forests model emerges as the best-performing model among the options provided, demonstrating its effectiveness in correctly identifying patterns among customers and their relationship with ABA's legal services, predicting the relationship between customers and ABA's legal services. One notable remark is that accuracy is lower than expected due to severely imbalanced data under different predictors, possibly a huge number of outliers & missing data.

Further studies on variable importance were conducted to see the top 3 most important predictors. Based on the analysis results, it was found that Annual Income, Allowed Income, and Marital Status emerged as the variables with the highest importance. This implies that these three predictors have a significant influence on the model's predictions and are crucial in determining the outcome or behavior being studied. Therefore, it is important to consider and give appropriate weightage to these variables when making predictions or drawing insights from the model.

  

### 5.  DISCUSSION
    

  

The exploratory data analysis provided valuable insights into the demographics of ABA's clients, including their geographic distribution, gender, race, and marital status. It also identified the most frequently asked categories and their association with specific states. This information can guide resource allocation and outreach strategies to effectively engage potential users and volunteers.

Based on our sentiment analysis, ABA should enhance support for the Family and Children category, address the high sadness sentiment in Health and Disability, investigate the elevated fear and anger in Individual Rights to improve customer satisfaction. Monitoring sentiment trends, fostering transparency, providing training, and collaborating with external organizations are recommended actions to ensure great services throughout the years.

Additionally, according to the outcomes of our trained machine learning model, ABA should consider Annual Income, Allowed Income, and Marital Status as key factors of higher importance when predicting a customer's inquiry category. Other combinations of selective variables of a prospective customer should also be taken into account. By leveraging this approach, ABA can assign specialized agents to assist customers, reducing human resource requirements and minimizing budgetary constraints. This enables the organization to allocate more focus and resources to the appropriate categories.

In summary, the findings from the exploratory data analysis and sentiment analysis provide valuable insights for ABA's strategy. By leveraging these insights, ABA can optimize resource allocation, improve customer satisfaction, and enhance overall service quality.
