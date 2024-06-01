
# This project is a collaboration with my friends in Big Data Analytics Project 
# Databricks notebook source
# MAGIC %fs ls /FileStore/tables/

# COMMAND ----------

pip install wordcloud

# COMMAND ----------

pip install textblob

# COMMAND ----------

# MAGIC %md
# MAGIC ## Data Pipeline

# COMMAND ----------

hotel_review = spark.read.csv('dbfs:/FileStore/tables/Datafiniti_Hotel_Reviews.csv',header=True)

# COMMAND ----------

print("Number of rows: %d" % hotel_review.count())
print("Number of columns: %d" % len(hotel_review.columns))

# COMMAND ----------

hotel_review.display()

# COMMAND ----------

new_cols = [c.replace(" ", "_").replace(",", "").replace("#","").replace(".","").replace("/","_") for c in hotel_review.columns]
hotel_review = hotel_review.toDF(*new_cols)

hotel_review.display()

# COMMAND ----------

from pyspark.sql.functions import *
onlyhotel=hotel_review.select('id','name','categories','longitude','latitude').distinct()
onlyhotel = onlyhotel.withColumn("latitude",col("latitude").cast("double")).withColumn("longitude",col("longitude").cast("double"))

# COMMAND ----------

from pyspark.sql.functions import split, trim,col
from pyspark.ml.feature import CountVectorizer
 
onlyhotel =onlyhotel.withColumn("desc_array",split(col("categories"),","))
onlyhotel1=onlyhotel.select([(col("desc_array")[x]).alias("Column"+str(x+1))
                   for x in range(0, 20)])
 

# COMMAND ----------

onlyhotel.display()

# COMMAND ----------

print("Number of hotels: %d" % onlyhotel.count())
print("Number of users: %d" % hotel_review.select("reviewsusername").distinct().count())

# COMMAND ----------

# MAGIC %md
# MAGIC ##Exploratory Data Analysis

# COMMAND ----------

# MAGIC %md
# MAGIC Check if the data contain missing values

# COMMAND ----------

hotel_review = hotel_review.select(['name','categories','primaryCategories','city','province','reviewsrating','reviewstext','reviewstitle'])
hotel_review.display()

# COMMAND ----------

from pyspark.sql import SparkSession
from pyspark.sql.functions import col, sum

# Count the number of missing values per column
missing_values_count = hotel_review.select([sum(col(column).isNull().cast("double")).alias(column) for column in hotel_review.columns])

# Print the results
missing_values_count.display()

# COMMAND ----------

# MAGIC %md
# MAGIC Reviews - City / Province

# COMMAND ----------

import pyspark.sql.functions as F
import seaborn as sns
import matplotlib.pyplot as plt

# count the number of reviews by city
city_counts = hotel_review.groupBy("city").agg(F.count("*").alias("count")).orderBy(F.desc("count")).limit(25)

# plot the bar chart for city counts
plt.figure(figsize=(12, 6))
sns.barplot(x="city", y="count", data=city_counts.toPandas())
plt.ylabel('The number of reviews per city')
plt.xlabel('City Name')
plt.xticks(rotation='vertical')
plt.show()

# COMMAND ----------

# count the number of reviews by province
province_counts = hotel_review.groupBy("province").agg(F.count("*").alias("count")).orderBy(F.desc("count"))

# plot the bar chart for province counts
plt.figure(figsize=(12, 6))
sns.barplot(x="province", y="count", data=province_counts.toPandas())
plt.ylabel('The number of reviews per province')
plt.xlabel('Province Code')
plt.xticks(rotation='vertical')
plt.show()

# COMMAND ----------

# MAGIC %md
# MAGIC #WordCloud 
# MAGIC We will now create a word cloud to see the most commonly occurring words in our reviews.

# COMMAND ----------

# MAGIC %md
# MAGIC Reviewstitle

# COMMAND ----------

from pyspark.sql import functions as F
review_title = hotel_review.select('reviewstitle').filter(F.col('reviewstitle').isNotNull())
review_title.display()

# COMMAND ----------

from pyspark.ml.feature import Tokenizer
tokenizer = Tokenizer(inputCol='reviewstitle',outputCol='review_terms')
tokenized_df = tokenizer.transform(review_title)
tokenized_df.display()

# COMMAND ----------

from pyspark.ml.feature import StopWordsRemover
stops_remover = StopWordsRemover(inputCol='review_terms',outputCol='review_nostops')
stops_df = stops_remover.transform(tokenized_df)
stops_df.display()

# COMMAND ----------

from pyspark.ml.feature import CountVectorizer
cv1 = CountVectorizer(inputCol='review_nostops',outputCol='features')
cv1_model = cv1.fit(stops_df)
cv1_df = cv1_model.transform(stops_df)
cv1_df.display()

# COMMAND ----------

#pip install wordcloud

# COMMAND ----------

from pyspark.ml.feature import CountVectorizer
from wordcloud import WordCloud
import matplotlib.pyplot as plt

# Get vocabulary
vocab = " ".join(cv1_model.vocabulary)

# Generate WordCloud
wordcloud = WordCloud(width = 800, height = 800,
                      background_color ='white',
                      min_font_size = 10).generate(vocab)

plt.figure(figsize = (8, 8), facecolor = None) 
plt.imshow(wordcloud, interpolation='bilinear') 
plt.axis("off") 
plt.title("Most common reviewstitle words", fontsize = 20)
plt.tight_layout(pad = 0) 
plt.show()

# COMMAND ----------

# MAGIC %md
# MAGIC Reviewstext

# COMMAND ----------

from pyspark.sql import functions as F
review_text = hotel_review.select('reviewstext').filter(F.col('reviewstext').isNotNull())
review_text.display()

# COMMAND ----------

from pyspark.ml.feature import Tokenizer
tokenizer = Tokenizer(inputCol='reviewstext',outputCol='review_terms')
tokenized_df = tokenizer.transform(review_text)
tokenized_df.display()

# COMMAND ----------

from pyspark.ml.feature import StopWordsRemover
stops_remover = StopWordsRemover(inputCol='review_terms',outputCol='review_nostops')
stops_df = stops_remover.transform(tokenized_df)
stops_df.display()

# COMMAND ----------

from pyspark.ml.feature import CountVectorizer
cv2 = CountVectorizer(inputCol='review_nostops',outputCol='features')
cv2_model = cv2.fit(stops_df)
cv2_df = cv2_model.transform(stops_df)
cv2_df.display()

# COMMAND ----------

from pyspark.ml.feature import CountVectorizer
from wordcloud import WordCloud
import matplotlib.pyplot as plt

# Get vocabulary
vocab = " ".join(cv2_model.vocabulary)

# Generate WordCloud
wordcloud = WordCloud(width = 800, height = 800,
                      background_color ='white',
                      min_font_size = 10).generate(vocab)

plt.figure(figsize = (8, 8), facecolor = None) 
plt.imshow(wordcloud, interpolation='bilinear') 
plt.axis("off") 
plt.tight_layout(pad = 0) 
plt.title("Most common reviewstext words", fontsize = 20)
plt.show()

# COMMAND ----------

# MAGIC %md
# MAGIC Get reviews polarity and subjectivity

# COMMAND ----------

from textblob import TextBlob
from pyspark.sql.functions import udf
from pyspark.sql.types import FloatType, StructType, StructField

# Define a UDF to get the polarity and subjectivity of a string
def get_sentiment(text):
    blob = TextBlob(text)
    polarity = blob.sentiment.polarity
    subjectivity = blob.sentiment.subjectivity
    return (polarity, subjectivity)

# Create a schema for the return type of the UDF
sentiment_schema = StructType([
    StructField("polarity", FloatType(), True),
    StructField("subjectivity", FloatType(), True)
])

# Create the UDF
get_sentiment_udf = udf(get_sentiment, sentiment_schema)

# COMMAND ----------

# Apply the UDF to the DataFrame
hotel_review = hotel_review.withColumn("reviewstext_sentiment", get_sentiment_udf("reviewstext"))

# Extract the polarity and subjectivity values into separate columns
hotel_review = hotel_review.withColumn("reviewstext_polarity", hotel_review["reviewstext_sentiment"]["polarity"])
hotel_review = hotel_review.withColumn("reviewstext_subjectivity", hotel_review["reviewstext_sentiment"]["subjectivity"])

# Drop the "sentiment" column
hotel_review = hotel_review.drop("reviewstext_sentiment")

# COMMAND ----------

hotel_review.display()

# COMMAND ----------

# MAGIC %md
# MAGIC **polarity with text weired**

# COMMAND ----------

weired=hotel_review.filter("reviewstext_polarity >0 and reviewsrating <3")
weired.display()

# COMMAND ----------

weired.collect()[67][6]

# COMMAND ----------

# MAGIC %md
# MAGIC Grouping dataframe by hotel's name, and get the average of each columns

# COMMAND ----------

hotel_review_grouped = hotel_review.groupby("name").mean()

# COMMAND ----------

hotel_review_grouped.display()

# COMMAND ----------

hotel_review.dtypes

# COMMAND ----------

hotel_review = hotel_review.withColumn("reviewsrating", col("reviewsrating").cast("double"))

# COMMAND ----------

grouped_df = hotel_review.groupBy("name").mean()
grouped_df.display()

# COMMAND ----------

from pyspark.sql.functions import count
reviews_count = hotel_review.groupBy("name").agg(count("reviewsrating").alias("total_reviews"))
reviews_count.display()

# COMMAND ----------

reviews_count.count()

# COMMAND ----------

hotel_review_grouped.count()

# COMMAND ----------

hotel_review = grouped_df.join(reviews_count,"name")

# COMMAND ----------

hotel_review.display()

# COMMAND ----------

# MAGIC %md
# MAGIC Binning, based on text polarity

# COMMAND ----------

from pyspark.sql.functions import udf
from pyspark.sql.types import StringType, FloatType

# Define the polarity_binning function
def polarity_binning(x):
    if x <= 0.0:
        return "Negative"
    else:
        return "Positive"

# Define a UDF (User-Defined Function) from the polarity_binning function
polarity_udf = udf(polarity_binning, StringType())

# Apply the polarity_udf to the 'polarity' column using a lambda function
hotel_review = hotel_review.withColumn("reviewstext_polarity_category", polarity_udf(hotel_review["avg(reviewstext_polarity)"]))

# Show the result
hotel_review.display()

# COMMAND ----------

# Group the data by polarity category and count the occurrences
hotel_review_grouped_textpolarity = hotel_review.groupBy("reviewstext_polarity_category").count()

# Convert the PySpark DataFrame to a Pandas DataFrame for plotting
hotel_review_grouped_textpolarity_pd = hotel_review_grouped_textpolarity.toPandas()

# Plot a countplot using Seaborn
sns.barplot(x="reviewstext_polarity_category", y="count",data=hotel_review_grouped_textpolarity_pd)
plt.title("Text review categories", fontsize=20)
plt.show()

# COMMAND ----------

from pyspark.sql.functions import udf
from pyspark.sql.types import StringType, FloatType

# Define the polarity_binning function
def polarity_binning_positive(x):
    if x >= 0.75 :
        return "Outstanding"
    elif x >= 0.4 :
        return "Great"
    elif x>0:
        return "Good"
    else :
        return "Negative"

# Define a UDF (User-Defined Function) from the polarity_binning function
polarity_positive_udf = udf(polarity_binning_positive, StringType())

# Apply the polarity_udf to the 'polarity' column using a lambda function
hotel_review = hotel_review.withColumn("reviewstext_polarity_positive_category", polarity_positive_udf(hotel_review["avg(reviewstext_polarity)"]))

# Show the result
hotel_review.display()

# COMMAND ----------

# Group the data by polarity category and count the occurrences
hotel_review_grouped_textpolarity_positive = hotel_review.groupBy("reviewstext_polarity_positive_category").count()

# Convert the PySpark DataFrame to a Pandas DataFrame for plotting
hotel_review_grouped_textpolarity_positive_pd = hotel_review_grouped_textpolarity_positive.toPandas()

# Plot a countplot using Seaborn
sns.barplot(x="reviewstext_polarity_positive_category", y="count",data=hotel_review_grouped_textpolarity_positive_pd)
plt.title("Text review categories", fontsize=20)
plt.show()

# COMMAND ----------

hotel_review_grouped_textpolarity_positive.display()

# COMMAND ----------

# MAGIC %md
# MAGIC create the first KPI

# COMMAND ----------

hotel_review = hotel_review.withColumn("KPI",hotel_review["avg(reviewstext_polarity)"]+hotel_review["avg(reviewsrating)"])

# COMMAND ----------

hotel_review.display()

# COMMAND ----------

# MAGIC %md
# MAGIC ## Category

# COMMAND ----------

# MAGIC %md
# MAGIC recommend top hotel in each category

# COMMAND ----------


selected=onlyhotel.filter("categories like '%,Spa%'")
selected.display()

# COMMAND ----------

selected1 = selected.join(hotel_review,"name")

# COMMAND ----------

from pyspark.sql.functions import desc

selected1 = selected1.orderBy(desc("KPI"))
selected1.display()

# COMMAND ----------

selected1 = selected1.withColumn("total_reviews", col("total_reviews").cast("double"))

# COMMAND ----------

selected1.dtypes

# COMMAND ----------

from pyspark.ml.linalg import Vectors,VectorUDT
from pyspark.sql.functions import udf



to_vector_udf = udf(lambda x: Vectors.dense(x), VectorUDT())


selected1 = selected1.withColumn("total_reviews_vector", to_vector_udf(selected1["total_reviews"]))



# COMMAND ----------

from pyspark.ml.feature import StandardScaler
from pyspark.ml.linalg import Vectors

# Create a StandardScaler object and fit it to the data
scaler = StandardScaler(inputCol="total_reviews_vector", outputCol="scaled_total_reviews",
                        withStd=True, withMean=True)
scalerModel = scaler.fit(selected1)

# Transform the data using the scaler
scaledData = scalerModel.transform(selected1)
scaledData.display()

# COMMAND ----------

def vector_to_float(v):
    return float(v[0])
udf_vector_to_float = udf(vector_to_float)
scaledData = scaledData.withColumn("scaled_total_reviews_float", udf_vector_to_float("scaled_total_reviews"))

# COMMAND ----------

scaledData.display()

# COMMAND ----------

# MAGIC %md
# MAGIC New KPI

# COMMAND ----------

scaledData = scaledData.withColumn("KPI2",scaledData["avg(reviewstext_polarity)"]+ scaledData["scaled_total_reviews_float"]+scaledData["avg(reviewsrating)"])
scaledData.display()

# COMMAND ----------

from pyspark.sql.functions import desc

scaledData = scaledData.orderBy(desc("KPI2"))
scaledData.display()

# COMMAND ----------

# MAGIC %md
# MAGIC City with casino hotel

# COMMAND ----------

hotel_review = spark.read.csv('dbfs:/FileStore/tables/Datafiniti_Hotel_Reviews.csv',header=True)
hotel_review.display()

# COMMAND ----------

casino_counts=hotel_review.select(['name','city','province']).filter("categories like '%,Casino%'").distinct()
casino_counts.display()


# COMMAND ----------

casino_counts=hotel_review.select('id','name','categories',"province").distinct()
casino_counts = casino_counts.filter(hotel_review["categories"].like("%Casino%")).groupBy("province").agg(F.count("*").alias("count")).orderBy(F.desc("count"))
casino_counts.display()

# COMMAND ----------

new_cols = [c.replace(" ", "_").replace(",", "").replace("#","").replace(".","").replace("/","_") for c in hotel_review.columns]
hotel_review = hotel_review.toDF(*new_cols)

hotel_review.display()
