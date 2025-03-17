import numpy as np
from typing import Any, ClassVar
from scipy.optimize import minimize
from features.prepare_for_training import prepare_for_training
from hypothesis.sigmoid import sigmoid


class logisticRegression:
    """
    手搓逻辑回归
    """

    def __init__(
        self,
        data,
        labels,
        polynomial_degree: int = 0,
        sinusoid_degree: int = 0,
        normalize_data: bool = True,
    ):
        (data_processed, feature_mean, feature_deviation) = prepare_for_training(
            data, polynomial_degree, sinusoid_degree, normalize_data
        )
        self.data = data_processed
        self.feature_mean = feature_mean
        self.feature_deviation = feature_deviation
        self.normalize_data = normalize_data
        self.sinusoid_degree = sinusoid_degree
        self.polynomial_degree = polynomial_degree
        self.num_features = self.data.shape[1]
        num_union_lables = np.unique(labels).shape[0]
        # 在分类中需要得到label中唯一值，即为结果值
        self.union_lables = np.unique(labels)

        self.labels = labels
        self.theta = np.zeros((num_union_lables, self.num_features))

    def train(self, max_iter: int = 500):
        cost_list = []
        for index, labal in enumerate(self.union_lables):
            current_theta = np.copy(self.theta[index].reshape(self.num_features))
            # 这里需要将 布尔矩阵转换为float
            current_labal = (self.labels == labal).astype(float)
            (res_theta, cost) = self.gradient_descent(
                self.data, current_labal, current_theta, max_iter
            )
            cost_list.append(cost)
            self.theta[index] = res_theta
        return self.theta, cost_list

    def gradient_descent(
        self, data: np.ndarray, labal: np.ndarray, theta: np.ndarray, max_iter: int
    ) -> list:
        cost_list = []
        res = minimize(
            lambda cur_theta: self.cost_fc(data, labal, cur_theta),
            theta,
            method="CG",
            jac=lambda cur_theta: logisticRegression.gradient_step(
                data, labal, cur_theta
            ),
            callback=lambda cur_theta: cost_list.append(
                self.cost_fc(data, labal, cur_theta)
            ),
            options={"maxiter": max_iter},
        )
        if not res.success:
            raise Exception("minimize failed")
        return res.x, cost_list

    def cost_fc(self, data: np.ndarray, labels: np.ndarray, theta: np.ndarray):
        m = data.shape[0]  # 获取样本数
        hypothesis = logisticRegression.hypothesis(data, theta)
        hypothesis = hypothesis.reshape(-1, 1)
        epsilon = 1e-15
        y_is_set = np.dot(
            labels[labels == 1].T, np.log(hypothesis[labels == 1] + epsilon)
        )
        y_is_not_set = np.dot(
            1 - labels[labels == 0].T, np.log(1 - hypothesis[labels == 0] + epsilon)
        )
        cost = (-1 / m) * (y_is_set + y_is_not_set)  # 分母修正为样本数
        return cost

    @staticmethod
    def gradient_step(data, labal, theta) -> None:
        num_features = data.shape[0]
        hypothesis = logisticRegression.hypothesis(data, theta)
        hy_diff = hypothesis - labal
        gredients = (1 / num_features) * np.dot(data.T, hy_diff)
        gredients = gredients.T.flatten()
        return gredients

    @staticmethod
    def hypothesis(data: np.ndarray, theta: np.ndarray):
        # 将预测值传入sigmoid中
        pre = sigmoid(np.dot(data, theta)).reshape(data.shape[0], 1)
        return pre

    @staticmethod
    def hypothesis2(data: np.ndarray, theta: np.ndarray):
        # 将预测值传入sigmoid中
        pre = sigmoid(np.dot(data, theta))
        return pre

    def get_cost(self, data: np.ndarray, labels):
        pass

    def predict(self, data: np.ndarray):
        (data_processed, feature_mean, feature_deviation) = prepare_for_training(
            data, self.polynomial_degree, self.sinusoid_degree, self.normalize_data
        )
        prob = logisticRegression.hypothesis2(data_processed, self.theta.T)
        max_prob_index = np.argmax(prob, axis=1)
        class_prediction = np.empty(max_prob_index.shape, dtype=object)
        for index, data in enumerate(self.union_lables):
            class_prediction[max_prob_index == index] = data
        return class_prediction
