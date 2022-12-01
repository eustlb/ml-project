from sklearn.metrics import r2_score

def eval_linear(y_true,y_pred):
    """
    Evaluate
    :param X:
    :param y:
    :return:
    """
    return r2_score(y_true, y_pred)




