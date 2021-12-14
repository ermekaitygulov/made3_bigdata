import numpy as np
import pandas as pd
import pytest
from pyspark.sql import SparkSession
from pyspark.sql import functions as SF

from linreg_spark import LinRegModel, LinRegEstimator


TARGET_WEIGHTS = np.array([1.5, 0.3, -0.7])


@pytest.fixture
def test_data():
    X = np.random.normal(size=(1000, 3))
    y = X @ TARGET_WEIGHTS[:, None]
    yield X, y


@pytest.fixture
def spark():
    yield SparkSession.builder.getOrCreate()


def test_lin_reg_model_predict_correct(spark):
    lr_model = LinRegModel(
        weights=[1., 2., 3.],
        bias=1.
    )
    lr_model = lr_model.setInputCol('features').setOutputCol('prediction')
    spark_data = spark.sparkContext.parallelize([[1, 1, 1]]).toDF()
    spark_data = spark_data.withColumn('features', SF.array('_1', '_2', '_3'))
    pred_data = lr_model.transform(spark_data).select('prediction').collect()[0].prediction
    assert pred_data == 7


def test_lin_reg_model_can_fit_test_data(test_data, spark):
    lr_estimator = LinRegEstimator(
        lr=1.,
        step=10
    )
    lr_estimator = (lr_estimator
                    .setInputCol('features')
                    .setOutputCol('prediction')
                    .setTargetCol('target'))
    X, y = test_data
    data = np.hstack([X, y])
    spark_data = spark.createDataFrame(pd.DataFrame(data, columns=['1', '2', '3', 'target']))
    spark_data = spark_data.withColumn('features', SF.array('1', '2', '3'))
    lr_model = lr_estimator.fit(spark_data)
    weights = lr_model.getWeights()
    bias = lr_model.getBias()
    assert all(np.isclose(weights, TARGET_WEIGHTS))
    assert np.isclose(bias, 0)


