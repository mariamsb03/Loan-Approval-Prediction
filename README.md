# Loan-Approval-Prediction
The aim of this project is predict if a person can get a loan or not based on a set of attributes. I used KNN and Logistic Regression to get the predictions. 
MACHINE LEARNING PROJECT 
Loan Approval Using KNN and Logistic Regression 
Faculuty of Information - Lebanese University 
Course: Machine Learning 
Doctor: Abbass Rammal 
Students: Lama Yaacoub & Mariam Sbeity 
Project Overview 
The project aimed to predict loan approval status based on various features such as gender, 
dependents, education, self-employment status, income, loan amount, loan term, credit history, 
and property area. Two machine learning models, K-Nearest Neighbors (KNN) and Logistic 
Regression, were implemented and evaluated for their effectiveness in predicting loan approval 
status. 
Dataset Description 
The dataset used for this project contains information about loan applicants, including their 
demographic and financial attributes, as well as the outcome variable indicating whether their loan 
application was approved or not. 
1 Loan : A unique id  
2 Gender : Gender of the applicant Male/female
3 Married : Marital Status of the applicant, values will be Yes/ No 
4 Dependents : It tells whether the applicant has any dependents or not. 


5 Education : It will tell us whether the applicant is Graduated or not. 
6 Self_Employed : This defines that the applicant is self-employed i.e. Yes/ No 
7 ApplicantIncome : Applicant income 
8 CoapplicantIncome : Co-applicant income 
9 LoanAmount : Loan amount (in thousands) 
10 Loan_Amount_Term Terms of loan (in months) 
11 Credit_History : Credit history of individual’s repayment of their debts 
12 Property_Area : Area of property i.e. Rural/Urban/Semi-urban  
13 Loan_Status : Status of Loan Approved or not i.e. Y- Yes, N-No 


K-Nearest Neighbors (KNN) 
What is knn? 
K-nearest neighbors (KNN) is a simple machine learning algorithm used for classification and 
regression tasks. It is a type of instance-based learning where the algorithm predicts the class of a 
new data point based on the majority class of its k nearest neighbors in the feature space. Is there 
anything else you would like to know about KNN? 
Purpose 
KNN is used in this project to classify loan applicants into approved or rejected categories based 
on the similarity of their features to those of previously observed applicants. It is particularly useful 
when the decision boundary between classes is nonlinear and when there is no clear separation 
between classes. 
How it works? 
Given k, the k-means algorithm is implemented in four steps: 
1. Partition objects into k nonempty subsets 
2. Compute centroids of the clusters of the current partitioning (the centroid is the center, i.e., 
mean point, of the cluster) 
3. Assign each object to the cluster with the nearest centroid point (according to a distance 
measure) 
4. Go back to Step 2, stop when the assignment does not change 
Logistic Regression  
What is Logistic Regression? 
It is a statistical method used for predicting the probability of a binary outcome based on one or 
more independent variables. Despite its name, logistic regression is primarily used for classification 
rather than regression tasks. 
Where: 
• P(Y=1∣X) is the probability of the positive class given the input features 𝑋X. 
• 𝛽β represents the coefficients of the logistic regression model. 
• 𝑋X represents the input features. 
Purpose 
Logistic Regression is used in this project to model the relationship between the input features 
and the probability of loan approval. By estimating the probability of loan approval for each 
applicant, logistic regression allows for the classification of applicants into approved or rejected 
categories based on a specified threshold. 
Contribution to the Project 
Logistic Regression provided insight into the factors influencing loan approval decisions and 
helped in identifying the most significant features affecting the outcome. By estimating the 
probabilities of loan approval, logistic regression allowed for more nuanced predictions compared 
to traditional binary classification algorithms. 
Hypothesis: 
H0: The hypothesis suggests that it can predict the likelihood of loan approval based on key 
features such as Marital Status, Education, Applicant Income, and Credit History, thereby 
streamlining the process for determining who qualifies for a loan 
Code: 
Step One: We imported the necessary libraries, and then we loaded the dataset.                     
• pandas (pd): Used for data handling and manipulation. 
• matplotlib. pyplot (plt),seaborn (sns): Enables data visualization. 
• sklearn. preprocessing: Preprocesses data for machine learning. 
• sklearn. model_selection.train_test_split: Splits data for training and testing. 
• sklearn. linear_model.LogisticRegression: Implements logistic regression for 
classification. 
• sklearn. neighbors. KNeighborsClassifier: Applies K-Nearest Neighbors algorithm for 
classification. 
• sklearn. metrics: Computes evaluation metrics for model performance. 
Step 2: Dataset Preprocessing and Visualizing 
1. Dataset Size: We use data. shape to understand the dataset's 
dimensions. 
2. Data Overview: data. info() gives a summary of the dataset's structure 
and types. 
3. Missing Values: data. isnull().sum() counts missing values in each 
column. 
4. Descriptive Statistics: data. describe () provides summary statistics 
for numerical columns. 
5. Removing 'Loan_ID': We drop the 'Loan_ID' column to remove 
unnecessary identifiers. 
First, we identified the categorical columns using data. dtypes == 'object'. Then, we extracted the 
names of these columns to create a list of object columns. With this list, we iterated over each 
object column. For each column, we counted the frequency of each category using 
data[col].value_counts (). These counts were then plotted as a bar plot using sns. barplot () to 
illustrate the distribution of categories. This visualization provided insights into the distribution of 
categorical variables in our dataset. 
We applied label encoding to categorical variables 
using `preprocessing.LabelEncoder()` to convert 
them into numeric values. Then, we filled missing 
values in the 'Credit_History' column with the mode 
value, which is the most frequent value. Afterward, 
we filled any remaining missing values in the dataset 
with the mean value of each respective column. 
Finally, we checked for any remaining missing values 
and ensured that the dataset was ready for further analysis by printing the updated null value 
counts and dataset information. 
Step 3: Data Analysis 
The output shows that after preprocessing the data, there are no 
missing values in any of the columns. All columns have 598 
non-null entries, indicating that the dataset is complete. 
Additionally, the data types have been appropriately converted 
for analysis. The 'Gender', 'Married', 'Education', 'Self_Employed', 
and 'Property_Area' columns have been encoded as integers, 
while the 'Dependents', 'CoapplicantIncome', 'LoanAmount', 
'Loan_Amount_Term', and 'Credit_History' columns remain as 
float64. Finally, the 'ApplicantIncome' column is of type int64. 
This confirms that the dataset is now ready for further analysis 
and model building. 
We calculated the correlation and we 
visualise it through heatmap, the most 
corelated columns appeard, Loan amount 
with application income: The correlation 
score was 0.52, which resembles a positive 
correlation, and Loan status with the credit 
history 0.54. 
After thecalculation of the correlation we 
visualise the corelation between the loan 
amount and the application income 
through a scatter plot. And it appeared 
postive and increasing correlation relation 
between these two columns 
This code snippet generates a visual representation 
of loan approval counts categorized by credit history 
using the seaborn library in Python. The seaborn 
countplot function is employed to plot the 
occurrences of different credit history values from the 
provided dataset, with loan approval status 
represented by color distinctions. The resulting plot 
offers insight into the distribution of loan approvals 
relative to varying credit histories, aiding in the 
analysis of how credit history influences loan 
approval decisions. Overall, this visualization 
facilitates a clearer understanding of the relationship 
between credit history and loan approval outcomes. 
Step 4: Splitting the Dataset and Implementing ML models 
In our code, we began by implementing logistic regression, a fundamental statistical method 
extensively used in binary classification tasks due to its simplicity and interpretability. Logistic 
regression estimates the probability of a binary outcome based on one or more predictor 
variables. After training the logistic regression model, we evaluated its performance using various 
metrics such as accuracy, precision, recall, and F1 score on both the training and testing datasets. 
These metrics provide insights into the model's ability to correctly classify loan applicants into 
approved or denied categories. Let's now delve into the results obtained from logistic regression 
and analyze its efficacy in predicting loan approval. 
We split our data into features (X) and the target variable (Y) representing loan status. Then, we 
divided the data into training and testing sets using the train_test_split function. After fitting the 
logistic regression model to the training data, we made predictions on both the training and 
testing sets. 
We calculated the confusion matrix for the testing set predictions and visualized it using a 
heatmap. The confusion matrix helps us understand the performance of our classifier by showing 
the counts of true positive, true negative, false positive, and false negative predictions. 
Finally, we evaluated the performance of our model using various metrics such as accuracy, 
precision, recall, and F1 score for both the training and testing sets. These metrics provide insights 
into how well the model performs in predicting loan approval status, considering both correct and 
incorrect predictions. 
1. Evaluation Metrics: - Accuracy: This metric indicates the overall correctness of the model's predictions. For both the 
training and testing sets, our logistic regression model achieves an accuracy of around 80%. This 
means that approximately 80% of the loan approval predictions made by the model are correct. - Precision: Precision measures the accuracy of positive predictions made by the model. In our 
case, it signifies the proportion of correctly predicted approved loans among all loans predicted as 
approved. Our model achieves a precision of approximately 79% on the training set and 81% on 
the testing set. - Recall: Recall, also known as sensitivity or true positive rate, indicates the proportion of actual 
positives that were correctly predicted by the model. It measures the ability of the model to 
identify all relevant instances. Our model demonstrates a recall of about 97% on the training set 
and 95% on the testing set. - F1 Score: The F1 score is the harmonic mean of precision and recall. It provides a balance 
between precision and recall. Our model achieves an F1 score of approximately 87% on both the 
training and testing sets.  
2.Confusion Matrix:   
The confusion matrix is a table that describes the performance of a classification model by 
comparing predicted labels with actual labels. In our confusion matrix: 
▪ The rows represent the actual loan approval status (0 for not approved, 1 for approved). 
▪ The columns represent the predicted loan approval status. 
▪ The numbers in each cell of the matrix indicate the counts of observations falling into each 
category. 
In our case: 
o There are 35 instances where the model correctly predicted loans as not approved 
(true negatives). 
o There are 38 instances where the model incorrectly predicted loans as approved 
when they were not (false positives). 
o There are 9 instances where the model incorrectly predicted loans as not approved 
when they were approved (false negatives). 
o There are 158 instances where the model correctly predicted loans as approved (true 
positives). 
In summary, these metrics and the confusion matrix provide a comprehensive understanding of 
how well our logistic regression model performs in predicting loan approval status, allowing us to 
assess its strengths and areas for improvement. 
Model 2: 
Following our exploration with logistic regression, we turned our attention to the K-Nearest 
Neighbors (KNN) algorithm, another widely used classification method. Unlike logistic regression, 
which is a parametric method, KNN is a non-parametric algorithm that makes predictions based 
on the majority class of the K nearest neighbors in the feature space. After training the KNN model 
with a specified number of neighbors, we assessed its performance using similar evaluation 
metrics as with logistic regression. The results obtained from KNN provide us with a different 
perspective on loan approval prediction, allowing us to compare and contrast its effectiveness with 
logistic regression. Let's now analyze the outcomes derived from the KNN algorithm and discuss 
its implications for our study. 
This code snippet involves training a k-nearest neighbors (KNN) classifier and evaluating its 
performance using various metrics, including the confusion matrix. 
1. Training the KNN Classifier: - We initialize a KNN classifier with n_neighbors=3, meaning it considers the 3 nearest 
neighbors when making predictions. - The classifier is then trained on the training data (X_train and Y_train) using the fit() method. 
2. Making Predictions: 
We use the trained classifier to make predictions on both the training set (X_train) and the testing 
set (X_test) using the predict() method. Predictions for both sets are stored in Y_train_pred and 
Y_test_pred, respectively. 
3. Confusion Matrix: - We calculate the confusion matrix using the confusion_matrix () function from scikit-learn. 
The confusion matrix provides a tabular representation of the model's performance by 
comparing predicted labels with actual labels. - The confusion matrix is then visualized using a heatmap generated by seaborn's heatmap () 
function. 
4. Evaluation Metrics: - Various evaluation metrics such as accuracy, precision, recall, and F1 score are computed for 
both the training and testing sets using scikit-learn's respective functions (accuracy_score, 
precision_score, recall_score, f1_score). - These metrics provide insights into the model's performance in terms of overall correctness, 
precision of positive predictions, ability to identify relevant instances, and balance between 
precision and recall. 
5. Printing Evaluation Metrics: - Finally, the evaluation metrics for both the training and testing sets are printed to the 
console, providing a quantitative assessment of the model's performance. - The metrics include accuracy, precision, recall, and F1 score for both sets. 
Overall, this code segment allows us to train a KNN classifier, assess its performance using the 
confusion matrix and various evaluation metrics, and compare its effectiveness on both the 
training and testing. This output represents the performance metrics of a KNN classifier after 
training and evaluation on both the training and testing datasets. 
Confusion Matrix: 
The confusion matrix provides a summary of the classifier's predictions compared to the actual 
labels in the testing set. It is a 2x2 matrix where each cell represents different combinations of true 
and predicted labels: - True Negative (TN): 19 cases where the classifier correctly predicted a negative outcome (0) 
when the actual label was also negative. - False Positive (FP): 54 cases where the classifier incorrectly predicted a positive outcome (1) 
when the actual label was negative. - False Negative (FN): 33 cases where the classifier incorrectly predicted a negative outcome 
(0) when the actual label was positive. - True Positive (TP): 134 cases where the classifier correctly predicted a positive outcome (1) 
when the actual label was also positive. 
Evaluation Metrics for Training Set: - Accuracy:78.49% - The proportion of correctly classified instances out of the total instances in 
the training set. - Precision: 79.72% - The proportion of correctly classified positive instances out of all instances 
classified as positive. 
- Recall: 91.80% - The proportion of correctly classified positive instances out of all actual positive 
instances. - F1 Score: 85.33% - The harmonic means of precision and recall, providing a balance between the 
two metrics. 
Evaluation Metrics for Testing Set: - Accuracy: 63.75% - The proportion of correctly classified instances out of the total instances in 
the testing set. - Precision: 71.28% - The proportion of correctly classified positive instances out of all instances 
classified as positive. - Recall: 80.24% - The proportion of correctly classified positive instances out of all actual positive 
instances. - F1 Score:75.49% - The harmonic means of precision and recall, providing a balance between the 
two metrics. 
These metrics offer insights into the classifier's performance in terms of overall correctness, its 
ability to identify relevant instances, and the balance between precision and recall. In this case, the 
model demonstrates relatively good performance on the training set, but it exhibits some drop in 
performance when applied to unseen data in the testing set, suggesting possible overfitting or 
generalization issues. 
Comparing the 2 Models: 
In comparing the performance of the logistic regression and k-nearest neighbors (KNN) models 
for our classification task, several key evaluation metrics were considered. The logistic regression 
model achieved higher accuracy scores on both the training and testing sets, indicating its 
superior ability to correctly classify instances. Furthermore, in terms of precision, the logistic 
regression model also outperformed KNN, demonstrating its capability to minimize false positives. 
Similarly, the logistic regression model exhibited higher recall rates, indicating its effectiveness in 
capturing positive instances from the total true positives. Additionally, when considering the F1 
score, which balances precision and recall, the logistic regression model consistently demonstrated 
better performance compared to KNN. Overall, these findings suggest that the logistic regression 
model offers more reliable and consistent predictions for our classification task, making it the 
preferred choice for this scenarion.                            
Thank you. 
