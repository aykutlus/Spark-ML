from pyspark.mllib.linalg import Vectors
from pyspark.sql import SparkSession

"""
+-----+-------------------+
|label|features           |
+-----+-------------------+
|0.0  | [1.0, 3.5, 4.2]   |
|1.0  | [5.2, -2.1, 3.5]  |
|0.0  | [-3, 2.4, 3.2]    |
|1.0  | [3.6, -5.0, 2.5]  |
+-----+-------------------+
"""

###################################### WAY 1 ##################################

# Set Spark session
spark = SparkSession \
    .builder \
    .getOrCreate()

# Prepare training data from a list of (label, features) tuples.
training = spark.createDataFrame([
    (0.0, Vectors.dense([1.0, 3.5, 4.2])),
    (1.0, Vectors.dense([5.2, -2.1, 3.5])),
    (0.0, Vectors.dense([-3, 2.4, 3.2])),
    (1.0, Vectors.dense([3.6, -5.0, 2.5]))], ["label", "features"])

print(training.show())


###################################### WAY 2 ##################################


from pyspark.ml.linalg import Vectors
from pyspark.ml.feature import VectorAssembler


dataset = [
    (0.0, Vectors.dense([1.0, 3.5, 4.2]),),
    (1.0, Vectors.dense([5.2, -2.1, 3.5]),),
    (0.0, Vectors.dense([-3, 2.4, 3.2]),),
    (1.0, Vectors.dense([3.6, -5.0, 2.5]),)]

df = spark.createDataFrame(dataset, ["label", "vector"])


assembler = VectorAssembler(
    inputCols=["vector"],
    outputCol="features")

output = assembler.transform(df)

output.select("label", "features").show(truncate=False)

###################################### WAY 3 ##################################



