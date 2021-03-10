import numpy as np 

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
    
