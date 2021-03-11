import numpy as np 
from collections import Counter

def accuracy(y_true,y_pred):
    correct_counter = 0
    for yt, yp in zip(y_true,y_pred):
        if yt == yp:
            correct_counter += 1
    return correct_counter/len(y_true)

#TP : True positive
#TN: True Negative
#FP: False Positive
#FN: False Negative
#Implementation to binary classification:

def true_positive(y_true, y_pred):
    tp = 0
    for yt,yp in zip(y_true,y_pred):
        if yt == 1 and yp == 1:
            tp+=1
    return tp

def true_negative(y_true, y_pred):
    tn = 0
    for yt,yp in zip(y_true,y_pred):
        if yt == 0 and yp == 0:
            tn+=1
    return tn

def false_positive(y_true, y_pred):
    fp = 0
    for yt,yp in zip(y_true,y_pred):
        if yt == 0 and yp == 1:
            fp+=1
    return fp

def false_negative(y_true, y_pred):
    fn = 0
    for yt,yp in zip(y_true,y_pred):
        if yt == 1 and yp == 0:
            fn+=1
    return fn

def accuracy_v2(y_true,y_pred):
    #obtain the same as accuracy using tp,tn,fp,fn
    tp = true_positive(y_true, y_pred)
    tn = true_negative(y_true, y_pred)
    fp = false_positive(y_true, y_pred)
    fn = false_negative(y_true, y_pred)
    acc_v2 = (tp+tn)/(tp+tn+fp+fn)
    return acc_v2

def precision(y_true,y_pred):
    #precision = TP/ (TP+FP)
    # If precision is low -> produce a lot of false positives -> bad
    tp = true_positive(y_true, y_pred)
    fp = false_positive(y_true, y_pred)
    prec = tp / (tp+fp)
    return prec

def recall(y_true,y_pred):
    #Recall = TP/ (TP+FN)
    # If recall is high -> produce less false negatives -> good 
    tp = true_positive(y_true, y_pred)
    fn = false_negative(y_true, y_pred)
    recall = tp / (tp+fn)
    return recall

# For models it's recommended high precision and recall
# However, it's difficult to select a ood treshold that gives both with high metrics.

#F1 is a metric that combines both, ideally its value is 1.
#F1 is used when datasets have skewed targets, instead of only accuracy.
def f1(y_true,y_pred):
    #F1 = 2PR/(P+R)
    # or F1 = 2TP/(2TP+FP+FN)
    p = precision(y_true,y_pred)
    r = recall(y_true,y_pred)
    f1_score = 2*p*r/(p+r)
    return f1_score

# TPR and FPR are useful to calculate ROC and ACU
#TPR : True Positive Rate
#FPR : False Positive Rate

def tpr(y_true,y_pred):
    # is the same as recall
    return recall(y_true,y_pred)

def fpr(y_true,y_pred):
    fp = false_positive(y_true, y_pred)
    tn = true_negative(y_true, y_pred)
    return fp/(tn+fp)

#Logloss

def log_loss(y_true,y_proba):
    # define an epsilon value
    # this can also be an input
    # this value is used to clip probabilities
    epsilon = 1e-15
    # initialize empty list to store
    # individual losses
    loss = []
    # loop over all true and predicted probability values
    for yt, yp in zip(y_true, y_proba):
        # adjust probability
        # 0 gets converted to 1e-15
        # 1 gets converted to 1-1e-15
        # Why? Think about it!
        yp = np.clip(yp, epsilon, 1 - epsilon)
        # calculate loss for one sample
        temp_loss = - 1.0 * ( yt * np.log(yp) + (1 - yt) * np.log(1 - yp))
        # add to loss list
        loss.append(temp_loss)
    # return mean loss over all samples
    return np.mean(loss)

#For multiclasss Classification:

def macro_precision(y_true,y_pred):
    num_classes = len(np.unique(y_true))
    # initialize precision to 0
    precision = 0
    # loop over all classes
    for class_ in range(num_classes):
        # all classes except current are considered negative
        temp_true = [1 if p == class_ else 0 for p in y_true]
        temp_pred = [1 if p == class_ else 0 for p in y_pred]
        # calculate true positive for current class
        tp = true_positive(temp_true, temp_pred)
        # calculate false positive for current class
        fp = false_positive(temp_true, temp_pred)
        # calculate precision for current class
        temp_precision = tp / (tp + fp)
        # keep adding precision for all classes
        precision += temp_precision
    # calculate and return average precision over all classes
    precision /= num_classes
    return precision

def micro_precision(y_true,y_pred):
    num_classes = len(np.unique(y_true))
    # initialize tp and fp to 0
    tp = 0
    fp = 0
    # loop over all classes
    for class_ in range(num_classes):
        # all classes except current are considered negative
        temp_true = [1 if p == class_ else 0 for p in y_true]
        temp_pred = [1 if p == class_ else 0 for p in y_pred]
        # calculate true positive for current class
        # and update overall tp
        tp += true_positive(temp_true, temp_pred)
        # calculate false positive for current class
        # and update overall tp
        fp += false_positive(temp_true, temp_pred)
    # calculate and return overall precision
    precision = tp / (tp + fp)
    return precision

def weighted_precision(y_true,y_pred):
    num_classes = len(np.unique(y_true))
    # create class:sample count dictionary
    # it looks something like this:
    # {0: 20, 1:15, 2:21}
    class_counts = Counter(y_true) # counts the ocurrences of each class in the data
    # initialize precision to 0
    precision = 0
    # loop over all classes
    for class_ in range(num_classes):
        # all classes except current are considered negative
        temp_true = [1 if p == class_ else 0 for p in y_true]
        temp_pred = [1 if p == class_ else 0 for p in y_pred]
        # calculate tp and fp for class
        tp = true_positive(temp_true, temp_pred)
        fp = false_positive(temp_true, temp_pred)
        # calculate precision of class
        temp_precision = tp / (tp + fp)
        # multiply precision with count of samples in class
        weighted_precision = class_counts[class_] * temp_precision
        # add to overall precision
        precision += weighted_precision
    # calculate overall precision by dividing by
    # total number of samples
    overall_precision = precision / len(y_true)
    return overall_precision

def weighted_f1(y_true,y_pred):
    num_classes = len(np.unique(y_true))
    # create class:sample count dictionary
    # it looks something like this:
    # {0: 20, 1:15, 2:21}
    class_counts = Counter(y_true)
    # initialize f1 to 0
    f1 = 0
    # loop over all classes
    for class_ in range(num_classes):
        # all classes except current are considered negative
        temp_true = [1 if p == class_ else 0 for p in y_true]
        temp_pred = [1 if p == class_ else 0 for p in y_pred]
        # calculate precision and recall for class
        p = precision(temp_true, temp_pred)
        r = recall(temp_true, temp_pred)
        # calculate f1 of class
        if p + r != 0:
            temp_f1 = 2 * p * r / (p + r)
        else:
            temp_f1 = 0  
        
        # multiply f1 with count of samples in class
        weighted_f1 = class_counts[class_] * temp_f1
        # add to f1 precision
        f1 += weighted_f1
        # calculate overall F1 by dividing by
        # total number of samples
    overall_f1 = f1 / len(y_true)
    return overall_f1

#Confusion matrix is a table that shows TP,FP,TN,FN

#For multilabel Classification for tasks like predict determined objects in a image, we
# will use other metrics like 
# Precision at K (P@k)
# Average precision at k (Ap@k)
# Mean average precision at k (MAP@k)
# Log loss

def pk(y_true, y_pred, k):
    """
    This function calculates precision at k
    for a single sample
    :param y_true: list of values, actual classes
    :param y_pred: list of values, predicted classes
    :param k: the value for k
    :return: precision at a given value k
    """
    # if k is 0, return 0. we should never have this
    # as k is always >= 1
    if k == 0:
        return 0
    # we are interested only in top-k predictions
    y_pred = y_pred[:k]
    # convert predictions to set
    pred_set = set(y_pred)
    # convert actual values to set
    true_set = set(y_true)
    # find common values
    common_values = pred_set.intersection(true_set)
    # return length of common values over k
    return len(common_values) / len(y_pred[:k])

def apk(y_true, y_pred, k):
    """
    This function calculates average precision at k
    for a single sample
    :param y_true: list of values, actual classes
    :param y_pred: list of values, predicted classes
    :return: average precision at a given value k
    """
    # initialize p@k list of values
    pk_values = []
    # loop over all k. from 1 to k + 1
    for i in range(1, k + 1):
        # calculate p@i and append to list
        pk_values.append(pk(y_true, y_pred, i))
    # if we have no values in the list, return 0
    if len(pk_values) == 0:
        return 0
    # else, we return the sum of list over length of list
    return sum(pk_values) / len(pk_values)

def mapk(y_true, y_pred, k):
    """
    This function calculates mean avg precision at k
    for a single sample
    :param y_true: list of values, actual classes
    :param y_pred: list of values, predicted classes
    :return: mean avg precision at a given value k
    """
    # initialize empty list for apk values
    apk_values = []
    # loop over all samples
    for i in range(len(y_true)):
        # store apk values for every sample
        apk_values.append(apk(y_true[i], y_pred[i], k=k) )
    # return mean of apk values list
    return sum(apk_values) / len(apk_values)

##Advanced metrics:
#QWK : quadratic weighted kappa or Cohen's kappa
#MCC :  Matthew's Correlation Coefficient [-1 1], 1 ok, -1 bad, 0 is random prediction

def mcc(y_true, y_pred):
    """
    This function calculates Matthew's Correlation Coefficient
    for binary classification.
    :param y_true: list of true values
    :param y_pred: list of predicted values
    :return: mcc score
    """
    tp = true_positive(y_true, y_pred)
    tn = true_negative(y_true, y_pred)
    fp = false_positive(y_true, y_pred)
    fn = false_negative(y_true, y_pred)

    numerator = (tp * tn) - (fp * fn)
    denominator =((tp + fp) * (fn + tn) * (fp + tn) * (tp + fn))
    denominator = denominator ** 0.5

    return numerator/denominator


###Regression metrics:
#Error = True Value - Predicted Value
#Absolute Error = abs(Error)
#Mean absolute error (MAE) = mean of absolute errors
#Squared error = Error ** 2
# MSE  Mean Squared Error = mean of Squared errors 
# RMSE root mean squared error = sqrt(MSE) -> most popular evaluating regression models
# Suqared logarithmic error SLE
# Mean SLE : MSLE
# Root MSLE: RMSLE

#Percentage Error = Error / TrueValue *100
#Mean Absolute Percentage Error: MAPE
# R**2 , R-Squared or coefficient of determination :0-1, could be negative for absurd predicitions
# 1 -> model fits data quite well

def mean_absolute_error(y_true, y_pred):
    """
    This function calculates mae
    :param y_true: list of real numbers, true values
    :param y_pred: list of real numbers, predicted values
    :return: mean absolute error
    """
    # initialize error at 0
    error = 0
    # loop over all samples in the true and predicted list
    for yt, yp in zip(y_true, y_pred):
        # calculate absolute error
        # and add to error
        error += np.abs(yt - yp)
    # return mean error
    return error / len(y_true)

def mean_squared_error(y_true, y_pred):
    """
    This function calculates mse
    :param y_true: list of real numbers, true values
    :param y_pred: list of real numbers, predicted values
    :return: mean squared error
    """
    # initialize error at 0
    error = 0
    # loop over all samples in the true and predicted list
    for yt, yp in zip(y_true, y_pred):
        # calculate squared error
        # and add to error
        error += (yt - yp) ** 2
    # return mean error
    return error / len(y_true)

def mean_squared_log_error(y_true, y_pred):
    """
    This function calculates msle
    :param y_true: list of real numbers, true values
    :param y_pred: list of real numbers, predicted values
    :return: mean squared logarithmic error
    """
    # initialize error at 0
    error = 0
    # loop over all samples in true and predicted list
    for yt, yp in zip(y_true, y_pred):
        # calculate squared log error
        # and add to error
        error += (np.log(1 + yt) - np.log(1 + yp)) ** 2
    # return mean error
    return error / len(y_true)

def mean_percentage_error(y_true, y_pred):
    """
    This function calculates mpe
    :param y_true: list of real numbers, true values
    :param y_pred: list of real numbers, predicted values
    :return: mean percentage error
    """
    # initialize error at 0
    error = 0
    # loop over all samples in true and predicted list
    for yt, yp in zip(y_true, y_pred):
        # calculate percentage error
        # and add to error
        error += (yt - yp) / yt
    # return mean percentage error
    return error / len(y_true)

def mean_abs_percentage_error(y_true, y_pred):
    """
    This function calculates MAPE
    :param y_true: list of real numbers, true values
    :param y_pred: list of real numbers, predicted values
    :return: mean absolute percentage error
    """
    # initialize error at 0
    error = 0
    # loop over all samples in true and predicted list
    for yt, yp in zip(y_true, y_pred):
        # calculate percentage error
        # and add to error
        error += np.abs(yt - yp) / yt
    # return mean percentage error
    return error / len(y_true)

def r2(y_true, y_pred):
    """
    This function calculates r-squared score
    :param y_true: list of real numbers, true values
    :param y_pred: list of real numbers, predicted values
    :return: r2 score
    """
    # calculate the mean value of true values
    mean_true_value = np.mean(y_true)
    # initialize numerator with 0
    numerator = 0
    # initialize denominator with 0
    denominator = 0
    # loop over all true and predicted values
    for yt, yp in zip(y_true, y_pred):
        # update numerator
        numerator += (yt - yp) ** 2
        # update denominator
        denominator += (yt - mean_true_value) ** 2
    # calculate the ratio
    ratio = numerator / denominator
    # return 1 - ratio
    return 1 – ratio


if __name__ == "__main__":
    y_train = [0,1,1,1,0,0,0,1]
    y_predict = [0,1,0,1,0,1,0,0]
    #print(accuracy(y_train,y_predict))
    #print(accuracy_v2(y_train,y_predict))
    #print(precision(y_train,y_predict))
    print(recall(y_train,y_predict))
    #using scikit-learn:
    #from sklearn import metrics
    #print(metrics.accuracy_score(y_train,y_predict))

    #Example2: considering a treshold which can change the recall , precision metrics drastically
    # y_pred represent the probability for a sample being assigned a value of 1, instead
    # of a defect treshold of 0.5  
    
