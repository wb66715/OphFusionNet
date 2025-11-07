import pandas as pd
from sklearn.metrics import confusion_matrix, roc_curve, precision_score, recall_score, f1_score
import seaborn as sns
from matplotlib import pyplot as plt
from sklearn.metrics import roc_auc_score
import numpy as np
from kappa import quadratic_weighted_kappa_from_matrix


# 这是一个多分类问题，y_true是target，y_pred是模型预测结果，数据格式为numpy

def cal(y_true, y_pred, y_pred_proba):
    # confusion matrix row means GT, column means predication
    cm = confusion_matrix(y_true, y_pred)
    '''画混淆矩阵'''
    #plt.rcParams['font.family'] = 'Times New Roman'

    plt.figure(figsize=(8, 7))

    index = [str(i) for i in range(cm.shape[0])]
    #index = ["dry", "normal", "pcv", "wet"]
    #index = ["normal", "early", "advanced"]
    #index = ["wet", "pcv", "dme"]
    #index = ["cscr", "dry", "mnv", "normal"]
    da = pd.DataFrame(cm, index=index)  # 转置输入数据
    sns.heatmap(da, annot=True, cbar=None, cmap='Blues', xticklabels=index, fmt='d',
                annot_kws={"size": 26})
    plt.xticks(fontsize=22)
    plt.yticks(fontsize=22)
    plt.tight_layout()  # 优化布局以减少空白区域
    plt.show()
    plt.close()


    # 计算总预测数量
    total_predictions = len(y_pred)

    # 初始化正确预测的计数器
    correct_predictions = 0

    # 遍历预测标签和真实标签，计算正确预测的数量
    for predicted, true in zip(y_pred, y_true):
        if predicted == true:
            correct_predictions += 1

    # 计算准确率
    accuracy = correct_predictions / total_predictions

    print("Accuracy:", accuracy)

    # 计算精确率、召回率、F1分数
    precision = precision_score(y_true, y_pred, average='macro', zero_division=0)
    recall = recall_score(y_true, y_pred, average='macro')
    f1 = f1_score(y_true, y_pred, average='macro', zero_division=0)

    # 计算Kappa系数
    kappa = quadratic_weighted_kappa_from_matrix(cm)

    sensitivity = []
    specificity = []
    for i in range(cm.shape[0]):  # 对每个类别
        tn = cm.sum() - cm[i, :].sum() - cm[:, i].sum() + cm[i, i]  # 总和减去该行和该列，再加上对角线
        fp = cm[:, i].sum() - cm[i, i]
        fn = cm[i, :].sum() - cm[i, i]
        tp = cm[i, i]

        sensitivity.append(tp / (tp + fn) if (tp + fn) > 0 else 0)
        specificity.append(tn / (tn + fp) if (tn + fp) > 0 else 0)
        # 计算平均敏感性和特异性
    sensitivity = sum(sensitivity) / len(sensitivity) if sensitivity else 0
    specificity = sum(specificity) / len(specificity) if specificity else 0

    print("精确率 (Precision): {:.3f}".format(precision))
    print("召回率 (Recall): {:.3}".format(recall))
    print("F1分数 (F1 Score): {:.3f}".format(f1))
    print("Kappa系数 (Cohen's Kappa): {:.4f}".format(kappa))
    print("敏感性 (Sensitivity): {:.3f}".format(sensitivity))
    print("特异性 (Specificity): {:.3f}".format(specificity))


    # 假设有四个类别的真实标签和预测概率
    y_true = y_true  # 真实标签
    y_pred_proba = np.array(y_pred_proba)  # 预测概率，形状为 (样本数量, 4)

    # 初始化存储每个二分类任务的 AUC 值的列表
    auc_scores = []

    # 初始化存储每个二分类任务的假阳率和真阳率的字典
    fpr_dict = {}
    tpr_dict = {}

    # 计算每个二分类任务的 AUC
    for i in range(cm.shape[0]):
        # 将当前类别视为正类，其他三个类别视为负类
        y_true_binary = [1 if label == i else 0 for label in y_true]
        y_pred_binary = y_pred_proba[:, i]

        # 计算 ROC 曲线
        fpr, tpr, _ = roc_curve(y_true_binary, y_pred_binary)

        # 计算 ROC 曲线下的面积 (AUC)
        auc = roc_auc_score(y_true_binary, y_pred_binary)

        # 将当前类别的 AUC 添加到列表中
        auc_scores.append(auc)

        # 存储假阳率和真阳率
        fpr_dict[i] = fpr
        tpr_dict[i] = tpr

    # 计算四分类任务的平均 AUC
    average_auc = sum(auc_scores) / len(auc_scores)
    print("AUC:", average_auc)

    # 绘制 ROC 曲线

    c = index
    plt.figure(figsize=(8, 7))
    for i in range(cm.shape[0]):
        plt.plot(fpr_dict[i], tpr_dict[i], label='{} (AUC = {:.2f})'.format(c[i], auc_scores[i]))
   
    plt.plot([0, 1], [0, 1], linestyle='--', color='gray')
    plt.xlabel('False Positive Rate', fontsize=22)
    plt.ylabel('True Positive Rate', fontsize=22)
    # plt.title('ROC Curve for Four-Class Classification')
    plt.legend(fontsize=22)
    plt.xticks(fontsize=22)
    plt.yticks(fontsize=22)
    plt.tight_layout()  # 优化布局以减少空白区域
    plt.grid(True)
    plt.show()





