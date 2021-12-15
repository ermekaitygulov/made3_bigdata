import numpy as np
from pyspark.ml import Estimator, Model
from pyspark.ml.util import DefaultParamsReadable, DefaultParamsWritable
from pyspark import keyword_only
from pyspark.sql.types import ArrayType, FloatType, StructType, StructField
from pyspark.sql.functions import udf
from pyspark.sql import functions as SF

from linreg_params import LinRegModelParams, LinRegEstimatorParams
from linreg_core import NumpyLinReg


def compute_grad_udf(x, y, weights, bias, lr=0.1):
    x = np.array([x])
    y = np.array(y)
    weights = np.array(weights)
    grad_w, grad_b = NumpyLinReg.compute_grad(x, y, weights, bias, lr)
    grad_w = grad_w.tolist()
    grad_b = grad_b.tolist()
    return [grad_w, grad_b]


compute_grad_udf = udf(compute_grad_udf, StructType([
    StructField('weights', ArrayType(FloatType())),
    StructField('bias', FloatType())
]))


class LinRegEstimator(Estimator, LinRegEstimatorParams, DefaultParamsReadable, DefaultParamsWritable):
    @keyword_only
    def __init__(self, lr=0.1, step=100):
        super(LinRegEstimator, self).__init__()
        kwargs = self._input_kwargs
        self.setParams(**kwargs)

    @keyword_only
    def setParams(self, lr=0.1, step=100):
        kwargs = self._input_kwargs
        return self._set(**kwargs)

    def _fit(self, dataset):
        x = self.getInputCol()
        y = self.getTargetCol()
        data_shape = dataset.count()
        x_example = dataset.select(x).limit(1).toPandas().values[0][0]

        weights_bias = NumpyLinReg.init_weights(np.array(x_example))
        for s in range(self.getStep()):
            grad_w_b = self.compute_grad_from_dataset(dataset, x, y, *weights_bias)
            weights_bias = NumpyLinReg.sgd_step(*grad_w_b, *weights_bias, data_shape)

        lr_model = self.init_model(*weights_bias)
        return lr_model

    def compute_grad_from_dataset(self, dataset, x, y, weights, bias):
        grad = dataset.select(self.compute_grad_expr(x, y, weights, bias).alias('grad'))
        grad = grad.select(
            *[SF.col('grad.weights').getItem(i).alias(f'w_{i}') for i, _ in enumerate(weights)],
            'grad.bias'
        )
        grad = grad.select(*[SF.sum(col) for col in grad.columns]).collect()[0]
        grad_w, grad_b = grad[:weights.shape[0]], grad[-1]
        grad_w = np.array(grad_w)[:, None]
        return grad_w, grad_b

    def compute_grad_expr(self, x, y, weights, bias):
        grad_expr = compute_grad_udf(
            SF.col(x), SF.col(y),
            SF.array(*[SF.lit(w.tolist()[0]) for w in weights]),
            SF.lit(bias),
            SF.lit(self.getLearningRate())
        )
        return grad_expr

    @staticmethod
    def init_model(weights, bias):
        weights = weights.squeeze().tolist()
        bias = bias
        lr_model = LinRegModel(
            weights=weights,
            bias=bias
        )
        return lr_model


class LinRegModel(Model, LinRegModelParams, DefaultParamsReadable, DefaultParamsWritable):
    @keyword_only
    def __init__(self, weights, bias):
        super(LinRegModel, self).__init__()
        kwargs = self._input_kwargs
        self.setParams(**kwargs)
        self.predict_udf = udf(self._predict, FloatType())

    @keyword_only
    def setParams(self, weights, bias):
        kwargs = self._input_kwargs
        return self._set(**kwargs)

    def _transform(self, dataset):
        x = self.getInputCol()
        y = self.getOutputCol()
        return dataset.withColumn(y, self.predict_udf(x))

    def _predict(self, x):
        weights = self.getWeights()
        bias = self.getBias()
        weights = np.array(weights)
        x = np.array(x)
        prediction = weights @ x + bias
        return prediction.tolist()


def to_udf(spark_type):
    def udf_wrapper(func):
        return udf(func, spark_type)
    return udf_wrapper

