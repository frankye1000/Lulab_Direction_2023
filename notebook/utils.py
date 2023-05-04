import seaborn as sns
import tensorflow as tf
import keras.backend as K
import matplotlib.pyplot as plt


def plt_roccurve(fpr,tpr,roc_auc):
    sns.set(font_scale=1.3)
    plt.figure()
    lw = 2
    plt.plot(
        fpr,
        tpr,
        color="darkorange",
        lw=lw,
        # label="ROC curve (area = %0.4f)" % roc_auc,
    )
    plt.plot([0, 1], [0, 1], color="navy", lw=lw, linestyle="--")
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    # plt.xlabel("False Positive Rate")
    # plt.ylabel("True Positive Rate")
    # plt.title("ROC curve")
    # plt.legend(loc="lower right")
    plt.show()



def plt_heatmap(con_ma):
    '''
    :input: con_ma:[[TP,FN],[FP,TN]]
    '''
    sns.set(font_scale=2)
    f,ax=plt.subplots()
    sns.heatmap(con_ma, annot=True, ax=ax, fmt='d', cmap='YlGnBu', vmax=200, vmin=0) # 畫熱力圖
    ax.set_title('confusion matrix') # 標題
    ax.set_xlabel('predict')         # x軸
    ax.set_ylabel('true')            # y軸
    ax.xaxis.set_ticklabels(['Down', 'Up'])
    ax.yaxis.set_ticklabels(['Down', 'Up'])
    plt.show()


def angle_difference(x, y):
    """
    Calculate minimum difference between two angles.
    """
    return abs(x - y)     ##　這邊要改！！！！！！！


def angle_error(y_true, y_pred):
    """
    Calculate the mean diference between the true angles
    and the predicted angles. Each angle is represented
    as a binary vector.
    """
    diff = angle_difference(K.argmax(y_true), K.argmax(y_pred))
    return K.mean(K.cast(K.abs(diff), K.floatx()))
