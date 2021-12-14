from pyspark.ml.param.shared import Params, TypeConverters, Param, HasInputCol, HasOutputCol


class HasTargetCol(Params):

    targetCol = Param(Params._dummy(), "targetCol", "target column name.", typeConverter=TypeConverters.toString)

    def __init__(self):
        super(HasTargetCol, self).__init__()

    def getTargetCol(self):
        return self.getOrDefault(self.targetCol)

    def setTargetCol(self, value):
        return self._set(targetCol=value)


class LinRegEstimatorParams(HasInputCol, HasOutputCol, HasTargetCol):

    lr = Param(
        Params._dummy(),
        "lr",
        "lr",
        typeConverter=TypeConverters.toFloat
    )
    step = Param(
        Params._dummy(),
        "step",
        "step",
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
