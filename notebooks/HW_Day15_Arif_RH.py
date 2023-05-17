#|----------------------------------------------------------------------------------|  
#  INFO
#|----------------------------------------------------------------------------------|  

### this rfm measured with reference date (2011-12-20)
### r_score,f_score,m_score measured by using quartile
### forked repo and assigment file https://github.com/arifalse/spark_docker/tree/main/notebooks

#|----------------------------------------------------------------------------------|  
#  Library
#|----------------------------------------------------------------------------------|

#### Library
import pyspark
import os
import json
import argparse
from dotenv import load_dotenv
from pathlib import Path
from pyspark.sql.types import StructType
from pyspark.sql import SparkSession, functions as F
from pyspark.sql.types import *
from datetime import datetime
import pandas as pd

#|----------------------------------------------------------------------------------|  
#  config path, env & sessions
#|----------------------------------------------------------------------------------|

#path resources
dotenv_path = Path('/resources/.env')

#credentials db
postgres_host = os.getenv('DIBIMBING_DE_POSTGRES_HOST')
postgres_db = os.getenv('DIBIMBING_DE_POSTGRES_DB')
postgres_user = os.getenv('DIBIMBING_DE_POSTGRES_ACCOUNT')
postgres_password = os.getenv('DIBIMBING_DE_POSTGRES_PASSWORD')

#spark context
sparkcontext = pyspark.SparkContext.getOrCreate(conf=(
        pyspark
        .SparkConf()
        .setAppName('Assignment_arif')
        .setMaster('local')
        .set("spark.jars", "/opt/postgresql-42.2.18.jar")
    ))
sparkcontext.setLogLevel("WARN")

spark = pyspark.sql.SparkSession(sparkcontext.getOrCreate())

#|----------------------------------------------------------------------------------|  
#   Functions
#|----------------------------------------------------------------------------------|

#### function to creat list of quartil frin df
def get_quantile(sdf,col) :
    df=sdf.toPandas()
    df=df[col].quantile([.25, .5, .75])
    return df.tolist()

#### function to score recency
def r_score(r_value,ls):
    ###recencty score calculatet by list of quantile of its value
    if r_value <= ls[0]:
        return 1
    elif r_value <= ls[1]:
        return 2
    elif r_value <= ls[2]:
        return 3
    else:
        return 4

#### function to score frequency
def f_score(f_value,ls):
    ###frecuency score calculatet by list of quantile of its value
    if f_value <= ls[0]:
        return 1
    elif f_value <= ls[1]:
        return 2
    elif f_value <= ls[2]:
        return 3
    else:
        return 4
      
#### function to score monetery
def m_score(m_score,ls):
    ###monetery score calculatet by list of quantile of its value
    if m_score <= ls[0]:
        return 1
    elif m_score <= ls[1]:
        return 2
    elif m_score <= ls[2]:
        return 3
    else:
        return 4

#### function to label rfm score
dict_label = {
  'Champion' : [444,443,433,434,343,344,334],
  'Loyal Customer' : [432,333,324,244,243,234,233,224],
  'Potential Loyalist': [442,440,441,430,431,422,421,420,341,340,331,330,320,342,322,321,312,242,241,240,231,230,222,212],
  'New Customer' : [401,400,311,421,412,300,200],
  'Promising' : [414,413,412,411,410,404,403,402,314,313,302,303,304,204,203,202],
  'Need Attention' : [424,423,332,323,232,223,214,213],
  'Cannot Lose Them' : [44,43,33,103,104,4,3,113],
  'About To Sleep' : [220,210,201,110,102],
  'At Risk' : [144,143,134,133,142,141,132,131,124,123,114,113,42,41,34,32,31,24,23,22,14,13],
  'Hibernating' : [221,211,120,130,140,122,121,112,111,21,12,11,101,100],
  'Lost' : [100] }

def rfm_label(x,dict_label) :
    val='Lost Customer'
    for key in dict_label :
        ls=dict_label.get(key)
        if int(x) in ls :
            val=key
    return val

#### register all function to udf
def r_udf(ls):
    return F.udf(lambda l: r_score(l, ls))
def f_udf(ls):
    return F.udf(lambda l: f_score(l, ls))
def m_udf(ls):
    return F.udf(lambda l: m_score(l, ls))
def rfm_udf(ls):
    return F.udf(lambda l: rfm_label(l, ls))

#|----------------------------------------------------------------------------------|  
#  Extract
#|----------------------------------------------------------------------------------|

jdbc_url = f'jdbc:postgresql://{postgres_host}/{postgres_db}'
jdbc_properties = {
    'user': postgres_user,
    'password': postgres_password,
    'driver': 'org.postgresql.Driver',
    'stringtype': 'unspecified'
}

sdf_retail = spark.read.jdbc(
    jdbc_url,
    'public.retail',
    properties=jdbc_properties
)

#|----------------------------------------------------------------------------------|  
#   Transforms
#|----------------------------------------------------------------------------------|

#get recency (now date - min date)
sdf_recency = sdf_retail\
    .groupBy('CustomerID')\
    .agg(F.datediff(F.lit(datetime(2011,12,20)),F.max('InvoiceDate')).alias('recency')).alias('sdf_recency')

#get frequency 
sdf_frequency = sdf_retail\
    .groupby('CustomerID')\
    .agg(F.countDistinct("InvoiceNo").alias("frequency")).alias('sdf_frequency')

#get monetery
sdf_monetery=sdf_retail\
    .groupby('CustomerID')\
    .agg(F.sum(F.col('UnitPrice')*F.col('Quantity')).alias('monetery')).alias('sdf_monetery')  

#join dataset recency frequency and monetery
join_1=sdf_recency\
        .join(sdf_monetery,sdf_recency.CustomerID==sdf_monetery.CustomerID,'inner')\
        .select(sdf_recency['*'],sdf_monetery.monetery)
sdf_cust=join_1\
        .join(sdf_frequency,join_1.CustomerID==sdf_frequency.CustomerID,'inner')\
        .select(join_1['*'],sdf_frequency.frequency)

#get rfm quartile and score it
quantile_r=get_quantile(sdf_recency,'recency')
quantile_f=get_quantile(sdf_frequency,'frequency')
quantile_m=get_quantile(sdf_monetery,'monetery')

sdf_cust=sdf_cust\
    .withColumn("r_score", r_udf(quantile_r)(F.col("recency")))\
    .withColumn("f_score", r_udf(quantile_r)(F.col("frequency")))\
    .withColumn("m_score", r_udf(quantile_r)(F.col("monetery")))

#add rfm score ana segment label
sdf_result=sdf_cust\
        .withColumn('rfm_score',F.concat(sdf_cust.r_score,sdf_cust.f_score,sdf_cust.m_score))\
        .withColumn("customer_segment", rfm_udf(dict_label)(F.col("rfm_score")))\
        .withColumn('date_ingest',F.lit(datetime.now()))

#check data result
sdf_result.show()

#|----------------------------------------------------------------------------------|  
#  Load
#|----------------------------------------------------------------------------------|

#### write to public.customer_segmentation_rfm
sdf_result\
    .write.format("jdbc")\
    .mode("overwrite")\
    .option("url", jdbc_url)\
    .option("dbtable", "public.customer_segmentation_rfm")\
    .option("user", jdbc_properties.get('user'))\
    .option("password", jdbc_properties.get('password'))\
    .option("driver", "org.postgresql.Driver")\
    .save()

#### test to read from public.customer_segmentation_rfm
sdf_segment = spark.read.jdbc(
    jdbc_url,
    'public.customer_segmentation_rfm',
    properties=jdbc_properties
)
sdf_segment.show()

