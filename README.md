# S7-CS4622-Machine_Learning-Lab1

## Introduction
In this report, I will discuss the various steps and techniques employed for feature selection and engineering on the AudioMNIST dataset. The dataset comprises two CSV files, 'train.csv' and 'valid.csv', containing 28,520 and 750 rows, respectively, with 256 features and 4 target labels each. The goal is to prepare a reduced set of features for predicting each of the four target labels: Speaker ID, Speaker Age, Speaker Gender, and Speaker Accent.

## Data Preprocessing
### Handling Null Values
The initial step in data preprocessing was to handle missing values. There were only missing values in ‘Label_2’

### Discretization of Age
Since there were relatively few unique values for speaker age, I chose to treat age as a discrete variable. It also helps when checking accuracy because I can use the same classification model for all labels.
### Feature Scaling
Normalization is a critical preprocessing step when working with SVC. I utilized the StandardScaler from scikit-learn to standardize. This process ensures that all features contribute equally to the model's training process and helps avoid issues with varying feature scales.

## Feature Selection and Removal
### Initial Attempts
To begin feature selection, I attempted to identify and remove irrelevant or redundant features. Two primary techniques were employed: correlation analysis and variance thresholding.

*Correlation Analysis:* I computed the pairwise correlation coefficients between all features. Features with low correlations with the target labels were candidates for removal. Surprisingly, no features were removed based on this criterion, suggesting that the dataset did not contain highly correlated or irrelevant features.

*Variance Thresholding:* I also considered using variance thresholding to remove features with low variance, as these may not carry much information. However, when applying a predefined threshold, no features were removed.

### SelectKBest and PCA

*SelectKBest:* I applied the SelectKBest method from the scikit-learn library. With k=100 as the number of selected features, this technique helped us identify the 100 most informative features for each target label. This method uses statistical tests to rank features based on their importance.

*Principal Component Analysis (PCA):* PCA is a dimensionality reduction technique that I used to extract the most significant components of the data while preserving a certain amount of variance. We configured PCA to retain 99% of the variance in the data ('n_components=0.99') and used the 'full' singular value decomposition (SVD) solver. This process resulted in 106 principal components for each target label, significantly reducing the dimensionality of the data.

PCA gave the highest accuracy.

## Model Selection and Training
For the classification tasks associated with the target labels (Speaker ID, Speaker Age, Speaker Gender, and Speaker Accent), I employed the Support Vector Classification (SVC) algorithm. SVC is a robust choice for classification tasks, and class_weightdict = ‘balanced’ for incorporating ‘label_4’ unbalanced distribution (it has nearly 70% of value 6).


## Conclusion
In conclusion, the feature selection and engineering process for the AudioMNIST dataset involved several steps and techniques. After handling missing values and discretizing the age variable, I attempted to remove irrelevant features using correlation analysis and variance thresholding, but no features met the criteria for removal.

To reduce dimensionality and extract the most informative features, I implemented SelectKBest and PCA. SelectKBest helped us select the top 100 features based on statistical tests, while PCA reduced the dimensionality to 106 components while retaining 99% of the variance.

PCA gave the highest accuracy. These reduced feature sets (106 features) were then used with the Support Vector Classification (SVC) algorithm for training and testing.

