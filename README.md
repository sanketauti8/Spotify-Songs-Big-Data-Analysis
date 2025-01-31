# Spotify-Songs-Big-Data-Analysis

This notebook processes and visualizes song data from a dataset. It performs various data analysis and visualization tasks such as:

Data Cleaning:
Removes the 'lyrics' column.
Drops missing values.
Converts certain columns (e.g., 'danceability', 'energy', 'tempo') to float.

Data Visualization:

Top Artists: Displays a bar chart of the top 10 artists based on the number of songs.

Top Songs by Tempo: Displays a bar chart of the top 10 songs by tempo.

Distribution of Features: Plots histograms for 'danceability', 'valence', 'energy', and others.

Loudness Distribution: Displays the distribution of loudness using a histogram with KDE.

Energy vs Danceability: Shows a hexbin plot to explore the relationship between energy and danceability.

Average Energy and Danceability by Tempo: Displays bar plots for average energy and danceability based on tempo categories.

Correlation Matrix: Visualizes the correlation between features using a heatmap.

Radar Chart: Plots a radar chart for a sample song showing values for features like danceability, energy, and loudness.

Filtering Songs:
Filters songs that don't have numbers in their names and are not too long.
Plots the top 10 songs based on different attributes (e.g., 'danceability', 'energy', 'loudness', etc.).

Saving Plots:
Saves each generated plot (e.g., plot1.png, plot2.png) to an S3 bucket using boto3.
