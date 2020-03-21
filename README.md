# Modeling-Cash-Withdrawals-From-ATMs

In this project my task was to predict number of cash withdrawals from an ATM by date and information given about the ATM (IDENTITY: each ATM has a different identity code, REGION: location of ATM, TRX_TYPE: corresponds to whether or not customers card is present, TRX_COUNT: number of cash withdrawals). I used two learners: Multilayer perceptron and random forest. I used these learners in order to predict future TRX_Count based on IDENTITY, REGION, DAY, MONTH, YEAR and TRX_type. I used Sklearn library to create and train learners. However, before training I had to preprocess data especially for Multilayer Perceptron. As preprocess, 
1) I optimized IDENTITY:
IDENTITY/ (IDENTITY.mean – IDENTITY.min)
2) Converted YEAR to 1 if 2019 else if 2018 0. (It was possible since whole data contains only two different years.) 
3) Converted TRX_type to 1 if 2 else if 1 0.
4) For Region, Month and Day; I used OneHotEncoder to convert them all to 1s and 0s. \n
After preprocessing my Data contained 66 fields: 31 for Day, 20 for Region, 12 for Month, 1 for Year, 1 for TRX_type and 1 for IDENTITY. 
Creating and training each learner was easy using Sklearn (2 lines each). The real challenge was to preprocess data according to present learners. For Random Forest: I used 200 different trees since error stops decreasing drastically after 200. For Multilayer Perceptron: I included 200 hidden layers because it is the default number, I used epsilon value of 1 since our values are in the range of 0 – 150, and a maximum iteration number of 500.
Random Forest RMSE before training whole data:  18.75603412138406
Multilayer Perceptron RMSE before training whole data:  15.535438482048463
This is the output I got from predicting test data for learners, which is randomly selected 10% of the total training data.
	I used Gradient Boosting Regressor for combining my learners. Sklearn includes Gradient Boosting Regressor in its tools so, it was also trivial to create and train. As X input for training, I combined each learners’ predictions as a NumPy array vertically and for y input the real labels. At first, I used the same data to train my ensembler that I used for training learners but after this training my ensembler was useless. So, I split training data into learner training data and ensembler training data (ensembler training data is 25% of training data). Finally, I reached: 
Gradient Boosting Regressor as Ensembler RMSE:  15.202752462939914

For the data contact: arasozkan576@gmail.com
