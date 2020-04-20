from pyspark.ml.classification import RandomForestClassifier
from pyspark.ml.evaluation import MulticlassClassificationEvaluator
from pyspark.sql import SparkSession

if __name__ == "__main__":


    spark_session = SparkSession\
        .builder\
        .appName("Spark Random Forest")\
        .master("local[5]")\
        .getOrCreate()

    data_frame = spark_session\
        .read\
        .format("libsvm")\
        .load("breast_cancer_scale.txt")

    data_frame.show(10)
    
    (training_data, test_data) = data_frame.randomSplit([0.72, 0.25])

    print("training data: " + str(training_data.count()))
    training_data.printSchema()
    training_data.show(5)

    print("test data: " + str(test_data.count()))
    test_data.printSchema()
    test_data.show(5)
    
    random_forest = RandomForestClassifier(featuresCol = 'features', labelCol = 'label')
    
    model = random_forest.fit(training_data)

    prediction = model.transform(test_data)
    
    prediction.printSchema()
    prediction.show(5)

    prediction\
        .select("prediction", "label", "features")\
        .show(5)
        
    # Select (prediction, true label) and compute test error
    evaluator = MulticlassClassificationEvaluator(
        labelCol="label", predictionCol="prediction", metricName="accuracy")
    
    accuracy = evaluator.evaluate(prediction)
    
    print("Test Accuracy = %g " % (accuracy))


    print("Test Error = %g " % (1.0 - accuracy))

    spark_session.stop()

