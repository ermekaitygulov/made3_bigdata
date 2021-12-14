import numpy as np
from pyspark.sql.functions import udf
from pyspark.sql.types import ArrayType, FloatType, StructType, StructField


def to_udf(spark_type):
    def udf_wrapper(func):
        return udf(func, spark_type)
    return udf_wrapper


class NumpyLinReg:
    @staticmethod
    def init_weights(X):
        dim = X.shape[1]
        weights = np.zeros((dim, 1))
        bias = 0
        return weights, bias

    @classmethod
    def compute_grad(cls, X, y, weights, bias, lr=0.1):
        err = cls.predict(X, weights, bias) - y
        grad_w = (err.T @ X).T
        grad_w *= lr
        grad_b = err.sum() * lr
        return grad_w, grad_b

    @staticmethod
    def sgd_step(grad_w, grad_b, weights, bias, batch_size):
        weights -= grad_w / batch_size
        bias -= grad_b / batch_size
        return weights, bias

    @staticmethod
    def predict(X, weights, bias):
        pred = X @ weights + bias
        return pred

    @classmethod
    @to_udf(StructType([
        StructField('weights', ArrayType(FloatType())),
        StructField('bias', FloatType())
    ]))
    def compute_grad_udf(cls, x, y, weights, bias, lr=0.1):
        x = np.array(x)
        y = np.array(y)
        grad_w, grad_b = cls.compute_grad(x, y, weights, bias, lr)
        grad_w = grad_w.tolist()
        grad_b = grad_b.tolist()
        return [grad_w, grad_b]




