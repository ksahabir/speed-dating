# speed-dating
Analysis and modeling of a dataset from an experiment with speed dating, with the aim to perform predictive modeling as well as some exploratory data analysis

In this project, I did some exploratory data analysis as well as classification utilizing a Naive
Bayes classifier that is specifically made for datasets with categorical variables.
The first thing I did was set na values in my dataset equal to the character “?”, as missing
values in the dataset were represented by the “?” character in the original dataset. I also made
use of the scikit.learn label encoding function to assign the string type variables as ordinal
numerical values. I mainly used this for the gender attribute, as that was the only variable with
string type variables that I included in the dataframe constructed after finding the correlation of
the columns in the original dataset. After editing the dataframe, I performed normalization on all
the columns because some columns had rankins with a range of 0-10 and others had a range of
0-5. Therefore, each column was normalized to fit a 0-1 range. This normalization, which was
min/max normalization, was performed by a function that was applied to each attribute and row
with the use of a “.apply” function.
The Naive Bayes classifier used was specifically designed for working with categorical data. All
that was needed was to find the general probabilities of a value belonging to a specific class in
the specified dataset. This is referred to as the prior probability. This results in two values, each
being a probability of a value belonging to either 0 or 1 in the “match” column (0 represented no
second date and 1 representing going on a second date). Then, we needed to find the
conditional probability of a value belonging to a specific attribute as well as a specific class. To
achieve this, we counted the number of values that belonged to each of the attributes and
divided them by the total number of values for each attribute per class. Finally, we multiplied the
prior probabilities by the conditional probabilities to get a dictionary of lists with 2 values (the
probability of each individual value belonging to each class). Then, the .argmax function was
used to select the highest probability and generate a list of predictions according to the class
with the highest probability per row in the testing dataframe. The accuracies ranged from 83% to
90% in the 10 times the classifier was run, which shows that the attributes selected in the
beginning were good at predicting whether or not the participants would go out on a second
date or not.
