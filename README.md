Netflix Content Trend Analysis — Project README

Project Overview:-
The Netflix Content Trend Analysis project provides a complete analytical overview of Netflix's content library. 
Using a dataset of Netflix titles, the project identifies trends in genres, content types, countries, durations, and ratings. 
It also generates automated summaries, visualizations, and a final PDF report to support data-driven decision-making.

Features:-
Data Cleaning & Preprocessing
Normalizes column names and handles missing values.
Extracts numeric duration from textual data.
Splits multiple genres and countries into separate lists for detailed analysis.
Extracts release year for time-based trends.
Content Type Analysis
Compares Movies vs TV Shows using bar and pie charts.
Generates yearly counts and stacked area charts for content growth trends.
Genre Analysis
Identifies top genres overall.
Tracks top genres over time to highlight emerging trends.
Generates bar plots, pie charts, line trends, and heatmaps.
Includes dendrogram to visualize genre similarity.
Country Analysis
Lists top countries by number of titles.
Creates bar charts and treemaps for visual comparison.
Shows country contributions over years.
Duration & Ratings Analysis
Distribution plots for numeric duration.
Count of seasons for TV shows.
Rating distribution analysis with top 20 ratings.
Automated Summary & Recommendations
Provides total counts of Movies, TV Shows, top genres, and top countries.
Highlights key insights and strategic recommendations for content investment.
PDF Report Generation
Consolidates all charts, tables, and insights into a single PDF report.
Includes early vs recent genre comparison to identify shifts in content trends.

Dataset:-
File: Netflix Dataset.csv
Format: CSV, containing information about Netflix titles, such as:
Title, Type, Genres, Country, Date Added, Duration, Rating
Place the dataset in the same folder as the script before running.

Installation & Requirements

Python Libraries Required:
pandas
numpy
matplotlib
seaborn
squarify
scipy
fpdf

Installation via pip:
pip install pandas numpy matplotlib seaborn squarify scipy fpdf

How to Run:- 
Place Netflix Dataset.csv in the same directory as the script.
Run the Python script:
python Netflix_Content_Trend_Analysis.py
Output will be saved in the outputs folder:
Figures: PNG charts for all visualizations.
Tables: CSV files summarizing analysis results.
PDF: Netflix_Content_Trends_Report.pdf consolidating all findings.
Automated Summary: automated_summary.txt for quick insights.

Output:-
Charts
Movies vs TV Shows: bar chart and pie chart
Top genres: bar chart and pie chart
Genre trends over time: line charts
Genre-year heatmap
Country contribution charts and treemap
Duration distribution and TV seasons
Ratings distribution
Tables

Summary CSVs: type counts, genre counts, country counts, pivot tables for year-wise analysis

PDF Report:-
Professional, consolidated report with all visualizations and insights
Includes strategic recommendations for content planning
Key Insights & Recommendations
Content Mix: Maintain balance between Movies and TV Shows based on trends.
Genre Focus: Invest in high-growth genres identified in the latest years.
Regional Strategy: Expand content in underrepresented countries showing rising trends.
Content Duration: Analyze viewer engagement with various durations and seasons.
Ratings Focus: Focus on producing content in higher-rated categories for audience retention.

Future Scope:-
Integrate viewership or engagement data to assess performance of genres/countries.
Include user review sentiment analysis to determine content quality perception.
Build an interactive dashboard for real-time Netflix content trend monitoring.
Implement predictive modeling to forecast future popular genres and release strategies.

Author:- 
Gundu Abhinav
Contact: abhinavsharmaa200510@gmail.com  
