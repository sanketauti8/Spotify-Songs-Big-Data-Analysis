# Databricks notebook source
from pyspark.sql import SparkSession
import seaborn as sns
#s3://hadoop-movie-analysis/Final_Hadoop_project_test.py
# create a SparkSession
#s3://hadoop-movie-analysis/Final_Hadoop_project.py
spark = SparkSession.builder.appName("myApp").getOrCreate()

# read CSV file from S3 bucket
songs_df = spark.read.format("csv").option("header", "true").load("S3 path of file")

# importing boto3 library to save the results in S3
from io import BytesIO
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages
import boto3

# make the connection to the S3 for storing your plots
s3 = boto3.client('s3')
buffer = BytesIO()
filenames = ['plot1.png', 'plot2.png', 'plot3.png','plot4.png','plot5.png','plot6.png','plot7.png','plot8.png','plot9.png','plot10.png','plot11.png']



# COMMAND ----------

# Dropping the 'lyrics' column

from pyspark.sql.functions import col
songs_df = songs_df.drop('lyrics')
songs_df = songs_df.dropna()


# List of columns to convert to float
columns_to_convert = ['danceability', 'energy', 'key', 'loudness', 'mode', 'speechiness',
                      'acousticness', 'instrumentalness', 'liveness', 'valence', 'tempo', 'duration_ms']

# Converting each column to float
for column in columns_to_convert:
    songs_df = songs_df.withColumn(column, col(column).cast("float"))



# COMMAND ----------

from pyspark.sql import functions as F
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# COMMAND ----------

# Most represented artists
top_artists_df = songs_df.groupBy('artists').count().orderBy(col('count').desc()).limit(10)

# Convert to Pandas for plotting
top_artists_pd = top_artists_df.toPandas()

# Plot
fig1 = plt.figure(figsize=(10, 6))
plt.bar(top_artists_pd['artists'], top_artists_pd['count'])
plt.xticks(rotation=90)
plt.title('Top 10 Artists by Number of Songs')
plt.xlabel('Artist')
plt.ylabel('Number of Songs')
plt.tight_layout()
plt.show()
fig1.savefig('plot1.png', dpi=300)

# COMMAND ----------

# Top songs based on their tempo
top_tempo_songs_df = songs_df.orderBy(col('tempo').desc()).limit(10)

# Convert to Pandas for plotting
top_tempo_songs_pd = top_tempo_songs_df.select('name', 'tempo').toPandas()

# Plot
fig2 = plt.figure(figsize=(10, 6))
plt.bar(top_tempo_songs_pd['name'], top_tempo_songs_pd['tempo'], color='orange')
plt.xticks(rotation=90)
plt.title('Top 10 Songs by Tempo')
plt.xlabel('Song Name')
plt.ylabel('Tempo (BPM)')
plt.tight_layout()
plt.show()
fig2.savefig('plot2.png', dpi=300)




# COMMAND ----------

# Difference of distribution of danceability, valence and energy

dance_valence_energy_pd = songs_df.select('danceability', 'valence', 'energy').toPandas()

# Plot distributions using histograms and KDE
fig3 = plt.figure(figsize=(18, 6))

# Plot 1: Danceability
plt.subplot(1, 3, 1)
sns.histplot(dance_valence_energy_pd['danceability'], bins=30, color='blue', kde=True)
plt.xlim(0, 1)  # Limit x-axis to the range 0 to 1
plt.title('Distribution of Danceability')
plt.xlabel('Danceability')
plt.ylabel('Count')

# Plot 2: Valence
plt.subplot(1, 3, 2)
sns.histplot(dance_valence_energy_pd['valence'], bins=30, color='green', kde=True)
plt.xlim(0, 1)  # Limit x-axis to the range 0 to 1
plt.title('Distribution of Valence')
plt.xlabel('Valence')
plt.ylabel('Count')

# Plot 3: Energy
plt.subplot(1, 3, 3)
sns.histplot(dance_valence_energy_pd['energy'], bins=30, color='red', kde=True)
plt.xlim(0, 1)  # Limit x-axis to the range 0 to 1
plt.title('Distribution of Energy')
plt.xlabel('Energy')
plt.ylabel('Count')

plt.tight_layout()
plt.show()
fig3.savefig('plot3.png', dpi=300)


# COMMAND ----------


loudness_pd = songs_df.select('loudness').toPandas()

# Plot loudness distribution with histogram + KDE
fig4 = plt.figure(figsize=(10, 6))

# histogram and KDE
sns.histplot(loudness_pd['loudness'], bins=30, color='purple', kde=True)


plt.title('Distribution of Loudness')
plt.xlabel('Loudness (dB)')
plt.ylabel('Count')


plt.tight_layout()
plt.show()
fig4.savefig('plot4.png', dpi=300)

# COMMAND ----------

#Co-relation between Energy and Danceability

energy_danceability_pd = songs_df.select('energy', 'danceability').toPandas()

# Hexbin plot for energy vs danceability
fig5 = plt.figure(figsize=(10, 6))
plt.hexbin(energy_danceability_pd['energy'], energy_danceability_pd['danceability'], gridsize=30, cmap='Blues', alpha=0.7)


plt.colorbar(label='Counts')

plt.title('Hexbin Plot: Energy vs Danceability')
plt.xlabel('Energy')
plt.ylabel('Danceability')
plt.xlim(0, 1)
plt.ylim(0, 1)
plt.tight_layout()
plt.show()
fig5.savefig('plot5.png', dpi=300)




# COMMAND ----------

#Average Energy and Danceability by Song Tempo
from pyspark.sql.functions import when

# Add a tempo category based on the BPM
songs_df = songs_df.withColumn('tempo_category', when(col('tempo') < 90, 'Low Tempo')
                                                  .when((col('tempo') >= 90) & (col('tempo') < 140), 'Medium Tempo')
                                                  .otherwise('High Tempo'))

# Calculate average energy and danceability per tempo category
tempo_avg_df = songs_df.groupBy('tempo_category').agg(F.avg('energy').alias('avg_energy'),
                                                      F.avg('danceability').alias('avg_danceability'))


tempo_avg_pd = tempo_avg_df.toPandas()

# Plot
fig6 = plt.figure(figsize=(10, 6))

plt.bar(tempo_avg_pd['tempo_category'], tempo_avg_pd['avg_energy'], alpha=0.7, label='Energy', color='blue')
plt.bar(tempo_avg_pd['tempo_category'], tempo_avg_pd['avg_danceability'], alpha=0.7, label='Danceability', color='green')

plt.title('Average Energy and Danceability by Tempo Category')
plt.xlabel('Tempo Category')
plt.ylabel('Average Value')
plt.legend()
plt.tight_layout()
plt.show()
fig6.savefig('plot6.png', dpi=300)


# COMMAND ----------

# Relation between every feature.

# Select relevant features for pair plot
features_pd = songs_df.select('energy', 'danceability', 'loudness', 'valence').toPandas()

# Create pair plot
sns.pairplot(features_pd)
plt.suptitle('Pair Plot of Selected Features', y=1.02)
plt.show()


# COMMAND ----------

# Check unique values in the mode column
mode_counts = songs_df.groupBy('mode').count().toPandas()
print(mode_counts)


if not mode_counts.empty:  
    fig7 = plt.figure(figsize=(10, 6))
    sns.boxplot(x='mode', y='loudness', data=songs_df.toPandas(), palette={'Major': 'green', 'Minor': 'blue'})
    plt.title('Loudness Distribution by Mode')
    plt.xlabel('Mode')
    plt.ylabel('Loudness (dB)')
    plt.tight_layout()
    plt.show()
    fig7.savefig('plot7.png', dpi=300)
else:
    print("No modes found in the dataset.")


# COMMAND ----------

# Calculate correlation matrix
corr = features_pd.corr()

# Heatmap of correlation matrix
fig8 = plt.figure(figsize=(8, 6))
sns.heatmap(corr, annot=True, cmap='coolwarm', square=True)
plt.title('Correlation Matrix Heatmap')
plt.tight_layout()
plt.show()
fig8.savefig('plot8.png', dpi=300)



# COMMAND ----------

import numpy as np

# Function to create radar chart
def create_radar_chart(data, title):
    labels = data.index
    stats = data.values
    angles = np.linspace(0, 2 * np.pi, len(labels), endpoint=False).tolist()

    stats = np.concatenate((stats,[stats[0]]))
    angles += angles[:1]

    fig, ax = plt.subplots(figsize=(6, 6), subplot_kw=dict(polar=True))
    ax.fill(angles, stats, color='blue', alpha=0.25)
    ax.plot(angles, stats, color='blue', linewidth=2)

    ax.set_yticklabels([])
    ax.set_xticks(angles[:-1])
    ax.set_xticklabels(labels)
    plt.title(title)
    plt.show()
    fig.savefig('plot9.png', dpi=300)

sample_song = songs_df.select('danceability', 'energy', 'loudness', 'valence').limit(1).toPandas().iloc[0]
create_radar_chart(sample_song, 'Radar Chart for Sample Song')


# COMMAND ----------

import re
import matplotlib.pyplot as plt
import seaborn as sns
from pyspark.sql import functions as F

# Function to filter songs that do not have numbers in their name and are not too long
def filter_songs(df, max_length=30):
    # Remove songs where the name contains numbers or is too long
    df_filtered = df.filter(~df['name'].rlike('\d'))  # Filter out songs with numbers
    df_filtered = df_filtered.filter(F.length(df['name']) <= max_length)  # Filter out long names
    return df_filtered

# Filter songs
max_length = 30
songs_filtered = filter_songs(songs_df, max_length)

# Function to plot top songs based on a given attribute
def plot_top_songs(df, attribute, title):
 
    df_pd = df.toPandas()
    
    fig9 = plt.figure(figsize=(12, 6))
    sns.barplot(x='name', y=attribute, data=df_pd, palette='viridis')
    plt.title(title)
    plt.xlabel('Song Name')
    plt.ylabel(attribute.capitalize())
    plt.xticks(rotation=45, ha='right')
    plt.tight_layout()
    plt.show()
    fig9.savefig('plot10.png', dpi=300)

# Get top 10 songs by danceability and plot
top_danceability = songs_filtered.orderBy(F.desc('danceability')).select('name', 'artists', 'danceability').limit(10)
plot_top_songs(top_danceability, 'danceability', 'Top 10 Songs by Danceability')

# Get top 10 songs by energy and plot
top_energy = songs_filtered.orderBy(F.desc('energy')).select('name', 'artists', 'energy').limit(10)
plot_top_songs(top_energy, 'energy', 'Top 10 Songs by Energy')

# Get top 10 songs by loudness and plot
top_loudness = songs_filtered.orderBy(F.desc('loudness')).select('name', 'artists', 'loudness').limit(10)
plot_top_songs(top_loudness, 'loudness', 'Top 10 Songs by Loudness')

# Get top 10 songs by valence and plot
top_valence = songs_filtered.orderBy(F.desc('valence')).select('name', 'artists', 'valence').limit(10)
plot_top_songs(top_valence, 'valence', 'Top 10 Songs by Valence')

# Get top 10 songs by tempo and plot
top_tempo = songs_filtered.orderBy(F.desc('tempo')).select('name', 'artists', 'tempo').limit(10)
plot_top_songs(top_tempo, 'tempo', 'Top 10 Songs by Tempo')


# COMMAND ----------

# Group by artist and find the maximum values for each attribute
artists_best = songs_filtered.groupBy('artists').agg(
    F.max('danceability').alias('max_danceability'),
    F.max('energy').alias('max_energy'),
    F.max('loudness').alias('max_loudness'),
    F.max('valence').alias('max_valence'),
    F.max('tempo').alias('max_tempo')
)

# Get top artists for danceability
top_danceability_artists = artists_best.orderBy(F.desc('max_danceability')).limit(10)

# Get top artists for energy
top_energy_artists = artists_best.orderBy(F.desc('max_energy')).limit(10)

# Get top artists for loudness
top_loudness_artists = artists_best.orderBy(F.desc('max_loudness')).limit(10)

# Get top artists for valence
top_valence_artists = artists_best.orderBy(F.desc('max_valence')).limit(10)

# Get top artists for tempo
top_tempo_artists = artists_best.orderBy(F.desc('max_tempo')).limit(10)

# Function to plot top artists for a given attribute
def plot_top_artists(df, attribute, title):
   
    df_pd = df.toPandas()
    
    fig11 = plt.figure(figsize=(12, 6))
    sns.barplot(x='artists', y=attribute, data=df_pd, palette='viridis')
    plt.title(title)
    plt.xlabel('Artist')
    plt.ylabel(attribute.capitalize())
    plt.xticks(rotation=45, ha='right')
    plt.tight_layout()
    plt.show()
    fig11.savefig('plot11.png', dpi=300)

# Plot top artists for each attribute
plot_top_artists(top_danceability_artists, 'max_danceability', 'Top 10 Artists by Danceability')
plot_top_artists(top_energy_artists, 'max_energy', 'Top 10 Artists by Energy')
plot_top_artists(top_loudness_artists, 'max_loudness', 'Top 10 Artists by Loudness')
plot_top_artists(top_valence_artists, 'max_valence', 'Top 10 Artists by Valence')
plot_top_artists(top_tempo_artists, 'max_tempo', 'Top 10 Artists by Tempo')


# COMMAND ----------

# Create a scatter plot of energy vs liveness
fig12 = plt.figure(figsize=(10, 6))
sns.scatterplot(x='liveness', y='energy', data=songs_df.toPandas(), alpha=0.5, color='purple')
plt.title('Energy vs Liveness')
plt.xlabel('Liveness')
plt.ylabel('Energy')
plt.tight_layout()
plt.show()
fig12.savefig('plot12.png', dpi=300)


# COMMAND ----------

import matplotlib.pyplot as plt
import pandas as pd


dance_party_songs_pd = dance_party_songs.toPandas()


fig13 = plt.figure(figsize=(12, 6))

# Create a bar plot for danceability and energy
plt.barh(dance_party_songs_pd['name'], dance_party_songs_pd['danceability'], color='blue', alpha=0.6, label='Danceability')
plt.barh(dance_party_songs_pd['name'], dance_party_songs_pd['energy'], color='orange', alpha=0.6, left=dance_party_songs_pd['danceability'], label='Energy')


plt.title('Top Dance Party Songs by Danceability and Energy')
plt.xlabel('Scores (0 to 1)')
plt.ylabel('Songs')
plt.legend()


plt.tight_layout()
plt.show()
fig13.savefig('plot13.png', dpi=300)


# COMMAND ----------

from pyspark.sql import functions as F
from pyspark.sql.functions import col

# Remove songs with numbers in their names
filtered_songs_df = songs_df.filter(~col('name').rlike('.*[0-9].*'))

# Peaceful songs: high valence, high acousticness, low energy
peaceful_songs = filtered_songs_df.filter((col('valence') > 0.7) & (col('acousticness') > 0.7) & (col('energy') < 0.3))

# Classical songs: high acousticness
classical_songs = filtered_songs_df.filter(col('acousticness') > 0.8)

# Uplifting songs: high valence and high energy
uplifting_songs = filtered_songs_df.filter((col('valence') > 0.7) & (col('energy') > 0.7))

# Selected relevant columns and limit the number of songs for each category
peaceful_songs = peaceful_songs.select('name', 'artists', 'valence', 'acousticness').orderBy(F.desc('valence')).limit(10)
classical_songs = classical_songs.select('name', 'artists', 'acousticness').orderBy(F.desc('acousticness')).limit(10)
uplifting_songs = uplifting_songs.select('name', 'artists', 'valence', 'energy').orderBy(F.desc('valence')).limit(10)


peaceful_songs_pd = peaceful_songs.toPandas()
classical_songs_pd = classical_songs.toPandas()
uplifting_songs_pd = uplifting_songs.toPandas()


def plot_songs(songs_df, title, color1, color2):
    fig14 = plt.figure(figsize=(12, 6))
    
   
    plt.barh(songs_df['name'], songs_df[color1], color='blue', alpha=0.6, label=color1)
    plt.barh(songs_df['name'], songs_df[color2], color='orange', alpha=0.6, left=songs_df[color1], label=color2)
    
   
    plt.title(title)
    plt.xlabel('Scores (0 to 1)')
    plt.ylabel('Songs')
    plt.legend()
    
 
    plt.tight_layout()
    plt.show()
    fig14.savefig('plot14.png', dpi=300)

# Plotting Peaceful Songs
plot_songs(peaceful_songs_pd, 'Top Peaceful Songs by Valence and Acousticness', 'valence', 'acousticness')

# Plotting Classical Songs (only acousticness)
fig15 = plt.figure(figsize=(10, 6))
plt.barh(classical_songs_pd['name'], classical_songs_pd['acousticness'], color='green', alpha=0.6)
plt.title('Top Classical Songs by Acousticness')
plt.xlabel('Acousticness')
plt.ylabel('Songs')
plt.tight_layout()
plt.show()
fig15.savefig('plot15.png', dpi=300)

# Plotting Uplifting Songs
plot_songs(uplifting_songs_pd, 'Top Uplifting Songs by Valence and Energy', 'valence', 'energy')

#Upload all plots to S3

for filename in filenames:
    with open(filename, 'rb') as file:
        s3.upload_fileobj(file, 'hadoop-spotify-analysis', filename)