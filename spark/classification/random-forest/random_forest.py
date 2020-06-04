import pyspark
from pyspark.sql import SparkSession
from pyspark.ml.evaluation import MulticlassClassificationEvaluator
from pyspark.mllib.util import MLUtils
from pyspark import SparkContext, SparkConf
from pyspark.ml.classification import RandomForestClassifier

spark = SparkSession.builder.appName('cruise').getOrCreate()

# Load and parse the data file into an RDD of LabeledPoint.
data = spark.read.csv('breast-cancer-wisconsin.csv', header = True, inferSchema = True)


from pyspark.ml.feature import VectorAssembler

assembler=VectorAssembler(inputCols=['clump_thickness','uniform_cell_size','uniform_cell_shape','marginal_adhesion','single_epi_cell_size','bland_chromation','normal_nucleoli','mitoses'],outputCol='features')
output=assembler.transform(data)
output.select('features','class').show(5)
#output as below
final_data=output.select('features','class')
#splitting data into train and test
train_data,test_data=final_data.randomSplit([0.7,0.3])
train_data.describe().show()



#the data into training and test sets (30% held out for testing)
# Train a DecisionTree model.
#  Empty categoricalFeaturesInfo indicates all features are continuous.

rf = RandomForestClassifier(labelCol="class", featuresCol="features")
# Fit the model
rfModel = rf.fit(train_data)

#dt = DecisionTreeClassifier(labelCol="class", featuresCol="features")
#model = dt.fit(train_data)

# Make predictions.
predictions = rfModel.transform(test_data)

# Select example rows to display.
predictions.select("prediction", "class", "features").show(5)

# Select (prediction, true label) and compute test error
evaluator = MulticlassClassificationEvaluator(
    labelCol="class", predictionCol="prediction", metricName="accuracy")
accuracy = evaluator.evaluate(predictions)
print ("Test Error = %g" % (1.0 - accuracy))
print(accuracy)
