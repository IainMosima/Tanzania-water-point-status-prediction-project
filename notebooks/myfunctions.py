# function to help with importing sets from thr analysis-df
def set_importer(path, y=False):
    import pandas as pd
    set = pd.read_csv(path)
    set.drop('Unnamed: 0', axis=1, inplace=True)

    if y == True:
        return set.squeeze()

    return set

#  function to help with printing score
def scores(y_true, y_preds, probs=None, print_results=True, log_loss=True):
    from sklearn.metrics import precision_score, recall_score, accuracy_score, auc, roc_curve, log_loss
    p_s = precision_score(y_true, y_preds)
    r_s = recall_score(y_true, y_preds)
    a_s = accuracy_score(y_true, y_preds)

    if print_results:
      print('The precision score is:\t', p_s)
      print('The recall score is:\t', r_s)
      print('The accuracy score is:\t', a_s)
   
    if log_loss:
        ll = log_loss(y_true, probs)
        fpr, tpr, thr =  roc_curve(y_true, probs)
        a_c = auc(fpr, tpr)
        
        if print_results:
            print('The log loss is:\t', ll)
            print('The auc is:\t', a_c)
    

    
    if print_results == False:
        return p_s, r_s, a_s, ll, a_c

        


