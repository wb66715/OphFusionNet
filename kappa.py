import numpy as np


def quadratic_weighted_kappa_from_matrix(confusion_matrix):
    # 将混淆矩阵转换为 numpy 数组
    cm = np.array(confusion_matrix)

    # 计算边际总和
    row_sums = cm.sum(axis=1)  # 每一行的总和
    col_sums = cm.sum(axis=0)  # 每一列的总和
    N = np.sum(row_sums)  # 样本总数

    # 计算期望频率矩阵
    expected_matrix = np.outer(row_sums, col_sums) / N

    # 计算类别数量
    num_classes = cm.shape[0]

    # 计算加权矩阵
    weights = np.zeros((num_classes, num_classes))
    for i in range(num_classes):
        for j in range(num_classes):
            weights[i, j] = 1 - (i - j) ** 2 / (num_classes - 1) ** 2

    # 计算观察到的一致性和期望一致性
    observed_agreement = np.sum(weights * cm) / N
    expected_agreement = np.sum(weights * expected_matrix) / N

    # 计算 QWK
    kappa = (observed_agreement - expected_agreement) / (1 - expected_agreement)

    return kappa

