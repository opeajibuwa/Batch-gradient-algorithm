import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from datetime import datetime
import seaborn as sns
import math


# Loads a data file from a provided file location.
def load_data(path):
    # Your code here:
    data = pd.read_csv(path)
    return data

# Implements dataset preprocessing, with boolean options to either normalize the data or not, 
# and to either drop the sqrt_living15 column or not.
#
# Note that you will call this function multiple times to generate dataset versions that are
# / aren't normalized, or versions that have / lack sqrt_living15.
def preprocess_data(data, normalize, drop_sqrt_living15):
    # Your code here:
    # (a) remove the id column
    data = data.iloc[:, 1:]
    #print("data after id column removal:", data.head(3), "\n")
    
    # (b) split the date feature into seperate columns
    data.date = pd.to_datetime(data.date)
    data['month'] = data['date'].dt.month
    data['year'] = data['date'].dt.year
    data['day'] = data['date'].dt.day
    data = data.iloc[:, 1:]
    #print("data after date column splitting:", data.head(3), "\n")
    
    # (c) create a dummy column of all ones as the bias/intercept term
    data['dummy'] = np.ones((data.shape[0], 1), dtype="int32")
    #print("data with dummy column inserted:", data.head(3))
    
    # (d) replace yr_renovated column with age_since_renovated column
    data.loc[data['yr_renovated'] == 0, 'age_since_renovated'] = data['year'] - data['yr_built']
    data.loc[data['yr_renovated'] != 0, 'age_since_renovated'] = data['year'] - data['yr_renovated']
    data = data.drop('yr_renovated', axis=1)
    all_columns = ['dummy', 'month', 'day', 'year', 'bedrooms', 'bathrooms', 'sqft_living', 'sqft_lot', 'floors', 
                       'waterfront', 'view', 'condition', 'grade', 'sqft_above', 'sqft_basement', 'yr_built', 
                       'age_since_renovated', 'zipcode', 'lat', 'long', 'sqft_living15', 'sqft_lot15', 'price']
    data = data[all_columns]
    #print("data after modifying yr-renovated column:", data.head(3), "\n")
    
    # split the data into X and y_labels
    X = data.drop("price", axis=1)
    y = data.price
    
    # (e) normalize dataset if normalize argument is set to true
    if normalize is True:
        column_mean = [0] * data.shape[1]
        column_std = [0] * data.shape[1]
        
        for index, column in enumerate(data.columns):
            if column == 'waterfront' or column == 'price' or column == 'dummy':
                continue
            column_mean[index] = data[column].mean()
            column_std[index] = data[column].std()
            data[column] = (data[column] - column_mean[index]) / column_std[index]
        
        if drop_sqrt_living15 is True:
            return data.drop("sqft_living15", axis=1), column_mean, column_std
        
        return data, column_mean, column_std      
        
    return data

# Implements the feature engineering required for part 4. Quite similar to preprocess_data.
# Expand the arguments of this function however you like to control which feature modification
# approaches are / aren't active.
def modify_features(data):
    # Your code here:

    return modified_data

# Trains a linear model on the provided data and labels, using the supplied learning rate.
# weights should store the per-feature weights of the learned linear regression.
# losses should store the sequence of MSE losses for each epoch of training, which you will then plot.
def gradient_train(X, y, lr, stopping_threshold, itr=float("inf")):
    # Your code here:
    counter = 0
    weight = np.zeros(X.shape[1])
    loss_list = [np.power(X.dot(weight) - y, 2).mean()]
    
    while counter < itr:
        delta_w = X.multiply((X.dot(weight) - y), axis=0).mean() * 2
        weight = weight - lr * delta_w
        loss = np.power(X.dot(weight) - y, 2).mean()
        loss_list.append(loss)
        
        if np.linalg.norm(delta_w) <= stopping_threshold or loss == np.nan or loss == float("inf"):
            break
            
        counter += 1
        
    print("itr={}, lr={}, loss={}".format(counter, lr, loss))
    return weight, loss_list

# Generates and saves plots of the training loss curves. Note that you can interpret losses as a matrix
# containing the losses of multiple training runs and then put multiple loss curves in a single plot.
def plot_losses(losses, lr, normalize):
    # Your code here:
    dict = {'iterations': list(range(len(losses))), 'mse': losses}
    df_iter_mse = pd.DataFrame(dict)
    
    plt.clf()
    iter_mse = sns.lineplot(data=df_iter_mse, x="iterations", y="mse").set_title('learning rate: ' + str(lr))
    fig_iter_mse = iter_mse.get_figure()
    
    if normalize:
        mse_plot_save_path = '/nfs/stak/users/ajibuwao/AI_534/figures/' + "normalized_lr_" + str(lr) + '.jpg'
        # mse_plot_save_path = '/Users/Opeyemi/Desktop/Machine Learning/homeworks/ia1/figures/' + "normalized_lr_" + str(lr) + '.jpg'
        fig_iter_mse.savefig(mse_plot_save_path)
    else:
        mse_plot_save_path = '/nfs/stak/users/ajibuwao/AI_534/figures/' + "non_normalized_lr_" + str(lr) + '.jpg'
        # mse_plot_save_path = '/Users/Opeyemi/Desktop/Machine Learning/homeworks/ia1/figures/' + "non_normalized_lr_" + str(lr) + '.jpg'
        fig_iter_mse.savefig(mse_plot_save_path)
    return

# compute the mse func
def compute_mse(X, y, weights):
    return np.power(X.dot(weights) - y, 2).mean()

# Invoke the above functions to implement the required functionality for each part of the assignment.

# Part 0  : Data preprocessing.
# Your code here:
# load in the training data
path = "/nfs/stak/users/ajibuwao/AI_534/IA1_train.csv"
# path = "/Users/Opeyemi/Desktop/Machine Learning/homeworks/ia1/IA1_train.csv"
data = load_data(path)
# load in the validation data
path = "/nfs/stak/users/ajibuwao/AI_534/IA1_dev.csv"
# path = "/Users/Opeyemi/Desktop/Machine Learning/homeworks/ia1/IA1_dev.csv"
dev = load_data(path)

# Part 1 . Implement batch gradient descent and experiment with different learning rates. 
""" Part (1a) -- Train the gradient descent algorithm using the preprocessed and normalized data"""
print("output for part 1(a):", "----------------------------------------------------------------------------")
train_data, column_mean, column_std = preprocess_data(data, normalize=True, drop_sqrt_living15=False)
lr_arr = [1e4, 1e3, 1e2, 1e1, 1e0, 1e-1, 1e-2, 1e-3, 1e-4, 1e-5]
for lr in lr_arr:
    weight, mse = gradient_train(train_data.drop('price', axis=1), train_data['price'], lr, 1e-8, itr=4000)
    plot_losses(mse, lr, normalize=True)

""" part 1(b) -- Compute and report the MSE of final model on the validation data"""
print("\n output for part 1(b):", "----------------------------------------------------------------------------")
dev_1 = preprocess_data(dev, normalize=False, drop_sqrt_living15=False)
# normalize the preprocessed dev data
for index, column in enumerate(dev_1.columns):
    if column == 'waterfront' or column == 'price' or column == 'dummy':
        continue
    dev_1[column] = (dev_1[column] - column_mean[index]) / column_std[index]
# report the mse of learned model on the validation data
convergent_lrs = [1e-1, 1e-2, 1e-3, 1e-4]
weight_list =  []
for lr in convergent_lrs:
    weight, mse = gradient_train(dev_1.drop('price', axis=1), dev_1['price'], lr, 1e-7, itr=4000)
    weight_list.append(weight)

for index, weight in enumerate(weight_list):
    print("learning rate={}, MSE={}".format(convergent_lrs[index], compute_mse(dev_1.drop("price", axis=1), dev_1['price'], weight)))

""" part 1(c) -- Report the learned weights for each feature of the model that performs best on the validation data"""
print("\n output for part 1(c):", "----------------------------------------------------------------------------")
# the best model has learning rate of 0.1
weight, mse = gradient_train(train_data.drop('price', axis=1), train_data['price'], 0.1, 1e-7, itr=4000)
print("learning rate={}, MSE={}".format(0.1, compute_mse(dev_1.drop("price", axis=1), dev_1['price'], weight)), "\n")

print(weight, "\n")
print("The most positive weight feature is {} and The most negative weight feature is {}".format(
     train_data.drop("price", axis=1).columns[np.argmax(weight)], train_data.drop("price", axis=1).columns[np.argmin(weight)]))

# Part 2 a. Training and experimenting with non-normalized data.
print("\n output for part 2(a):", "----------------------------------------------------------------------------")
# fetch the non-normalized data
train_data2 = preprocess_data(data, normalize=False, drop_sqrt_living15=False)
# train the non-normalized data with different learning rates
lr_arr = [10, 1, 1e-1, 1e-2, 1e-3, 1e-4, 1e-5, 1e-6, 1e-8, 1e-9, 1e-10, 1e-11, 1e-12, 1e-13, 1e-14, 1e-15, 1e-16]
for lr in lr_arr:
    weight, mse = gradient_train(train_data2.drop('price', axis=1), train_data2['price'], lr, 1e-7, itr=4000)
    plot_losses(mse, lr, normalize=False)

# compute the MSE of the best model on the validation dataset
dev_2 = preprocess_data(dev, normalize=False, drop_sqrt_living15=False)
# evaluate the best learned model on the validation dataset
# the best model has learning rate of 1e-12
weight, mse = gradient_train(train_data2.drop('price', axis=1), train_data['price'], 1e-12, 1e-7, itr=4000)
print("learning rate={}, MSE={}".format(1e-12, compute_mse(dev_2.drop("price", axis=1), dev_2['price'], weight)), "\n")
print(weight, "\n")
print("The most positive weight feature is {} and The most negative weight feature is {}".format(
     train_data2.drop("price", axis=1).columns[np.argmax(weight)], train_data2.drop("price", axis=1).columns[np.argmin(weight)]))


# Part 2 b Training with redundant feature removed.
print("\n output for part 2(b):", "----------------------------------------------------------------------------")
# fetch the normalized data with the 'sqft_living15' column dropped
train_data3, column_mean3, column_std3 = preprocess_data(data, normalize=True, drop_sqrt_living15=True)

#train the model with the new dataset and compute the validation mse and the learned weights
weight, mse = gradient_train(train_data3.drop('price', axis=1), train_data3['price'], 0.1, 1e-7, itr=4000)
print("learning rate={}, MSE={}".format(0.1, compute_mse(dev_1.drop(['price', 'sqft_living15'], axis=1), dev_1['price'], weight)), "\n")
print(weight)




