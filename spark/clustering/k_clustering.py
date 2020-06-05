import pyspark
from pyspark.sql import SparkSession
from pyspark.ml.evaluation import MulticlassClassificationEvaluator


spark = SparkSession.builder.appName('k-mean').getOrCreate()

# Loads data.
df = spark.read.csv('breas-cancer.csv', header = True, inferSchema = True)


from pyspark.ml.feature import VectorAssembler

vecAssembler = VectorAssembler(inputCols=['clump_thickness','uniform_cell_size','uniform_cell_shape','marginal_adhesion','single_epi_cell_size','bland_chromation','normal_nucleoli','mitoses'], outputCol="features")
new_df = vecAssembler.transform(df)
new_df.show()

from pyspark.ml.clustering import KMeans

kmeans = KMeans(k=2, seed=1)  # 2 clusters here
model = kmeans.fit(new_df.select('features'))

transformed = model.transform(new_df)
transformed.show()
