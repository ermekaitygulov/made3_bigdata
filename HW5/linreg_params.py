from pyspark.ml.param.shared import Params, TypeConverters, Param, HasInputCol, HasOutputCol


class LinRegEstimatorParams(HasInputCol, HasOutputCol):

    lr = Param(
        Params._dummy(),
        "learning_rate",
        "learning_rate",
        typeConverter=TypeConverters.toFloat
    )
    step = Param(
        Params._dummy(),
        "sgd steps",
        "sgd steps",
        typeConverter=TypeConverters.toInt
    )

    def __init__(self):
        super(LinRegEstimatorParams, self).__init__()

    def setInputCol(self, value):
        return self._set(inputCol=value)

    def setOutputCol(self, value):
        return self._set(outputCol=value)

    def setLearningRate(self, value):
        return self._set(lr=value)

    def getLearningRate(self):
        return self.getOrDefault(self.lr)

    def setStep(self, value):
        return self._set(step=value)

    def getStep(self):
        return self.getOrDefault(self.step)


class LinRegModelParams(HasInputCol, HasOutputCol):

    weights = Param(
        Params._dummy(),
        "weights",
        "weights",
        typeConverter=TypeConverters.toVector
    )
    bias = Param(
        Params._dummy(),
        "bias",
        "bias",
        typeConverter=TypeConverters.toFloat
    )

    def __init__(self):
        super(LinRegModelParams, self).__init__()

    def setInputCol(self, value):
        return self._set(inputCol=value)

    def setOutputCol(self, value):
        return self._set(outputCol=value)

    def setWeights(self, value):
        return self._set(weights=value)

    def getWeights(self):
        return self.getOrDefault(self.weights)

    def setBias(self, value):
        return self._set(bias=value)

    def getBias(self):
        return self.getOrDefault(self.bias)
