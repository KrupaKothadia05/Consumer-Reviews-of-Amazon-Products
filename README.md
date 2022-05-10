# Consumer-Reviews-of-Amazon-Products
Amazon is one of the largest companies across the world spreaded in various different departments such as E-commerce, cloud computing, AI/ ML, Digital streaming and much more. Millions of people buy products from amazon and share their sentiments for the product in the form of reviews. 

In this project we will be analyzing the Electronics products that are sold on amazon e-commerce website to find the best selling product of amazon from customer perspective, know the customer satisfaction, pricing and much more. 

Natural Language Processing (NLP) is a branch of computer science that studies the interactions between humans and computers. Sentiment analysis, or identifying a sentence as good or negative, is one of NLP's subproblems. Classifying a statement as positive or negative by using various ML models is proven beneficial to reduce the cost and time of man hours working on identifying the reviews given by users that whether it is a good or bad review. 

We have used a dataset of Amazon’s Consumer Reviews of Amazon Products (https://www.kaggle.com/datasets/datafiniti/consumer-reviews-of-amazon-products) which is from www.kaggle.com and have done sentiment analysis on the dataset using various ML models.
To increase the efficiency of our predicted outcomes we implemented various different algorithms with transformation methods such as Count Vectorization and TF_IDF vectorization.

Algorithms:
Logistic Regression: This algorithm is generally used for classification problems and also when there are more than one independent attribute. Also explains the correlation between dependent and independent variables.
KNN Classifier: KNN stands for K nearest neighbor. As the name suggests in this algorithm neighbors n have the crucial role to play. This algorithm is quite slow as compared to other algorithms.
AdaBoost Classifier: As the name suggests Ada boost algorithms increase the speed of algorithm and increase efficiency. The AdaBoost algorithm makes decisions using a bunch of decision stumps. The tree is then tweaked to focus where the prediction were incorrect
XGBoost: This algorithm basically boosts the speed and efficiency of the framework. 
XGBoost with hyper parameters tuning: We initially did not think of implementing this algorithm but since we got good results with XGboost that has default parameters we tried using hyperparameters since we can get much better results by tuning or adjusting the parameters.
Gaussian Naive Bayes: Gaussian Naive Bayes is used whenever all the continuous variables with each feature are distributed according to Gaussian distribution.
Decision Tree classifier: This algorithm can handle both numerical and categorical data. This algorithm requires a minimum amount of data preparation and can also handle multiple output problems.
Random Forest Algorithm: Random forest algorithm can handle large datasets efficiently. This algorithm also produces good precautions and is easy to understand.

Workflow: 

For coding purposes, we have used Python, Jupyter Notebook, Pandas, Numpy, Matplotlib, Scikit-learn, Seaborn, Vectorization, wordcloud, corpus etc. Then we are importing our dataset from Kaggle and performing some analysis using various data visualization plots. After analyzing the data we are processing it for sentiment analysis and then we are doing a train test split of the dataset. Next step is to train the ML models on a train data set to predict the sentiments of the user for a given product. For evaluation purposes we are using a test set from the train test split we did before and using accuracy score to determine the accuracy of ML algorithms for predicting the sentiments from texts given by users. 

Visualization:

Matrix plot: Data sparsity across all dataframe columns is depicted by a missingno matrix plot. We can check for missing values in our dataset using the matrix plot below.


Here, since we have no values in reviews.didPurchase, reviews.id, reviews.userCity, and reviews.userProvince columns, we have removed all the three rows from our dataset. We can see here that the dataset is quite large and very few null values, so can drop the null rows. 

Count Plot: The countplot is used to represent the occurrence(counts) of the observation present in the categorical variable. We may check for any data imbalance using the count plot below. The first chart depicts the ratings given by the users starting from 1-5, second we have a brand section where all the products are from Amazon only. Lastly we have a recommender graph which depicts how many users recommend the product to buy. 

Histogram Plot: A histogram is the graphical representation of data where data is grouped into continuous number ranges and each range corresponds to a vertical bar.

Sentiment Analysis: In our dataset, we are creating one new column named “sentiments” which is used for sentiment analysis and it is derived from reviews.rating column where if the value in reviews.rating column is less than 4 then sentiments column will have zero value stating that the user is not satisfied with the product and if it is greater than 4 then it will be 1 meaning the customer is happy/satisfied with the product. Since we are using this new column named sentiments for our sentiment analysis we are dropping the column review.rating from our dataset.

Pie Chart: Pie charts are useful for presenting a parts-to-whole connection for categorical or nominal data. The pie chart below shows the percentage sentiment distributions.

Wordcloud: A word cloud is a graphic depiction of a text in which words become larger as they are referenced more frequently. Word clouds are an excellent way to visualize unstructured text data and to see trends and patterns. The wordcloud below is of reviews.text field, which shows that there are a very high number of positive reviews.

Experiments  / Proof of concept evaluation:

Source of Dataset: We have used a dataset of Amazon’s Consumer Reviews of Amazon Products which is from www.kaggle.com. The dataset has more than 34,000 reviews for amazon products such as kindle, Fire stick TV and more, with each product having 21 features such as: id, name, asins, brand, categories, keys, manufacturing, review date, reviews.id, reviews.username etc. 
	
Data preprocessing decisions:
So as mentioned above the missing values part remains the same. Now we can see  reviews.text has certain words that wont be useful for prediction and will create a mess instead. These words are called Stopwords which need to be removed from prediction in order to get good results. So will implement a corpus, filter the text, remove punctuation and will create a new column that will have sentences without stopwords. After that we will convert the text into numeric form using vectorization methods since machine learning algorithms do not accept string data type.

Methodology:
When done with the pre-processing data we split the dataset into parts: training set and testing set. The training set will have a review_text column and the testing set has sentiments stored in it. Later on we have implemented various algorithms to predict the best outcomes.

