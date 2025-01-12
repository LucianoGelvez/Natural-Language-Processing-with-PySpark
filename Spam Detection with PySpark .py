from pyspark.sql import SparkSession
from pyspark.sql.functions import length
from pyspark.ml.feature import Tokenizer, StopWordsRemover, CountVectorizer, IDF, VectorAssembler, StringIndexer
from pyspark.ml.classification import LogisticRegression
from pyspark.ml import Pipeline
from pyspark.ml.evaluation import MulticlassClassificationEvaluator

# Initialize Spark Session
spark = SparkSession.builder.appName("spam_detection").getOrCreate()

# Load Dataset
data = spark.read.csv("smsspamcollection/SMSSpamCollection", inferSchema=True, sep='\t')

# Rename Columns
data = data.withColumnRenamed('_c0','class').withColumnRenamed('_c1','text')

# Feature Engineering
data = data.withColumn('length', length(data['text']))

# Feature Transformation stages
tokenizer = Tokenizer(inputCol='text', outputCol='token_text')
stop_remove = StopWordsRemover(inputCol='token_text', outputCol='stop_token')
count_vec = CountVectorizer(inputCol='stop_token', outputCol='c_vec')
idf = IDF(inputCol='c_vec', outputCol='tf_idf')
ham_spam_to_numeric = StringIndexer(inputCol='class', outputCol='label')
clean_up = VectorAssembler(inputCols=['tf_idf', 'length'], outputCol='features')

# Pipeline
data_prep_pipe = Pipeline(stages = [ham_spam_to_numeric, tokenizer, stop_remove, count_vec, idf, clean_up])
cleaner = data_prep_pipe.fit(data)
cleaner_data = cleaner.transform(data)

# Data splitting and model training
training, test = cleaner_data.randomSplit([0.7, 0.3])
spam_detector = LogisticRegression().fit(training)

# Model predictions
test_result = spam_detector.transform(test)


# Model evaluation
acc_eval = MulticlassClassificationEvaluator()
acc = acc_eval.evaluate(test_result)

# Print Results
print("Accuracy:", acc)
test_result.show()

spark.stop()