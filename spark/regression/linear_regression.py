
# make pyspark importable as a regular library.
import findspark

# create a SparkSession
from pyspark.sql import SparkSession
spark = SparkSession.builder.getOrCreate()
# load data
data = spark.read.csv('./boston_housing.csv', header=True, inferSchema=True)
# create features vector
print(data.head(5))

feature_columns = data.columns[:-1]
 # here we omit the final column


from pyspark.ml.feature import VectorAssembler
assembler = VectorAssembler(inputCols=feature_columns,outputCol="features")
data_2 = assembler.transform(data)
# train/test split
train, test = data_2.randomSplit([0.7, 0.3])


# define the model
from pyspark.ml.regression import LinearRegression

algo = LinearRegression(featuresCol="features", labelCol="medv")
# train the model

model = algo.fit(train)

# evaluation
evaluation_summary = model.evaluate(test)
evaluation_summary.meanAbsoluteError
evaluation_summary.rootMeanSquaredError
evaluation_summary.r2
# predicting values
predictions = model.transform(test)
predictions.select(predictions.columns[13:]).show() # here I am filtering out some columns just for the figure to fit
