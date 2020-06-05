from pyspark.sql import SparkSession
from pyspark.ml.recommendation import ALS
from pyspark.ml.evaluation import RegressionEvaluator

spark = SparkSession.builder.appName('recommender').getOrCreate()
df = spark.read.csv('ratings.csv', inferSchema= True, header = True)

df.show(3)

df.describe().show()

train, test = df.randomSplit([0.8, 0.2])

als = ALS(maxIter=5, regParam=0.01, userCol='userId', itemCol='movieId', ratingCol='rating')

model = als.fit(train)
predictions = model.transform(test)
predictions.show()

evaluator = RegressionEvaluator(metricName = 'rmse', labelCol = 'rating', predictionCol = 'prediction')
rmse = evaluator.evaluate(predictions)
print('RMSE:', rmse)

this_user = test.filter(test['userId'] == 12).select('userId', 'movieId')
this_user.show()

recommendation_this_user = model.transform(this_user)
recommendation_this_user.show()

recommendation_this_user.orderBy('prediction', ascending=False).show()
