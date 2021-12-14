import numpy as np
from pyspark.ml import Estimator, Model
from pyspark.ml.util import DefaultParamsReadable, DefaultParamsWritable
from pyspark import keyword_only
from pyspark.sql.types import FloatType
from pyspark.sql.functions import udf

from linreg_params import LinRegModelParams, LinRegEstimatorParams
from linreg_core import NumpyLinReg


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
        x_example = dataset.select(x).limit(2).toPandas().values

        weights_bias = NumpyLinReg.init_weights(x_example)
        for s in range(self.getStep()):
            grad = dataset.select(NumpyLinReg.compute_grad_udf(x, y, *weights_bias, self.getLearningRate()))
            grad = grad.groupBy().sum().collect()[0][0]
            weights_bias = NumpyLinReg.sgd_step(*grad, *weights_bias, data_shape)

        weights, bias = weights_bias
        weights = weights.tolist()
        bias = bias.tolist()
        lr_model = LinRegModel(weights, bias)
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

