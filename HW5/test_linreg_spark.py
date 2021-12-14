import numpy as np
import pandas as pd
import pytest
from pyspark.sql import SparkSession
from pyspark.sql import functions as SF

from linreg_spark import LinRegModel


@pytest.fixture
def test_data():
    weights = np.array([[1.5], [0.3], [-0.7]])
    X = np.random.normal(size=(1000, 3))
    y = X @ weights
    yield X, y


@pytest.fixture
def spark():
    yield SparkSession.builder.getOrCreate()


def test_lin_reg_model_predict_correct(test_data, spark):
    lr_model = LinRegModel(
        weights=[1., 2., 3.],
        bias=1.
    )
    lr_model = lr_model.setInputCol('features').setOutputCol('prediction')
    spark_data = spark.sparkContext.parallelize([[1, 1, 1]]).toDF()
    spark_data = spark_data.withColumn('features', SF.array('_1', '_2', '_3'))
    pred_data = lr_model.transform(spark_data).select('prediction').collect()[0].prediction
    assert pred_data == 7


