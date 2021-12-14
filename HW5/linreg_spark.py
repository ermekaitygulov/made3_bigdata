import numpy as np
from pyspark.ml import Estimator, Model
from pyspark.ml.util import DefaultParamsReadable, DefaultParamsWritable
from pyspark import keyword_only
from pyspark.sql.types import FloatType
from pyspark.sql.functions import udf

from linreg_params import LinRegModelParams


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

