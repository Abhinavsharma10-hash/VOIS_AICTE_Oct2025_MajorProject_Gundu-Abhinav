"""
Netflix Content Trend Analysis — Full Report Generator
Saves: PNG charts, CSV summary tables, and a consolidated PDF report.
Place 'Netflix Dataset.csv' in the same folder as this script.
"""

import os
import textwrap
from datetime import datetime
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import squarify            # treemap
from scipy.cluster.hierarchy import linkage, dendrogram
from fpdf import FPDF

# -----------------------------
# Settings & Output folders
# -----------------------------
DATA_FILE = 'Netflix Dataset.csv'   # filename in same folder
OUT_DIR = 'outputs'
FIG_DIR = os.path.join(OUT_DIR, 'figures')
TABLE_DIR = os.path.join(OUT_DIR, 'tables')
PDF_FILE = os.path.join(OUT_DIR, 'Netflix_Content_Trends_Report.pdf')

for d in (OUT_DIR, FIG_DIR, TABLE_DIR):
    os.makedirs(d, exist_ok=True)

sns.set(style='whitegrid')
plt.rcParams.update({'figure.max_open_warning': 0})

# -----------------------------
# 1) Load and normalize dataset
# -----------------------------
if not os.path.exists(DATA_FILE):
    raise FileNotFoundError(f"Place your dataset file named '{DATA_FILE}' in this folder and re-run the script.")

df = pd.read_csv(DATA_FILE)
# Normalize column names
df.columns = df.columns.str.strip().str.lower().str.replace(' ', '_')

# Show basic info
print("Loaded dataset:", DATA_FILE, "shape:", df.shape)
print(df.columns.tolist())

# Identify plausible columns
title_col = 'title' if 'title' in df.columns else df.columns[0]
country_col = 'country' if 'country' in df.columns else None
genre_col = None
for c in ['listed_in', 'category', 'genres', 'genre']:
    if c in df.columns:
        genre_col = c
        break
type_col = 'type' if 'type' in df.columns else None
release_col = None
for c in ['date_added', 'release_date', 'release_date_obj', 'release_year']:
    if c in df.columns:
        release_col = c
        break
duration_col = 'duration' if 'duration' in df.columns else None
rating_col = 'rating' if 'rating' in df.columns else None

# -----------------------------
# 2) Preprocessing
# -----------------------------
# Safe conversions and fills
if release_col:
    df[release_col] = pd.to_datetime(df[release_col], errors='coerce')
    df['year'] = df[release_col].dt.year
else:
    # try to find a numeric year column
    for c in df.columns:
        if df[c].dtype in [np.int64, np.float64] and ((df[c] > 1900) & (df[c] < 2100)).any():
            df['year'] = df[c].astype('Int64')
            release_col = c
            break
    if 'year' not in df.columns:
        df['year'] = np.nan

if genre_col:
    df[genre_col] = df[genre_col].fillna('Unknown')
    df['genres_list'] = df[genre_col].astype(str).str.split(',')
    df['genres_list'] = df['genres_list'].apply(lambda x: [g.strip() for g in x] if isinstance(x, list) else [])
else:
    df['genres_list'] = [[]]

if country_col:
    df[country_col] = df[country_col].fillna('Unknown')
    df['country_list'] = df[country_col].astype(str).str.split(',')
    df['country_list'] = df['country_list'].apply(lambda x: [c.strip() for c in x] if isinstance(x, list) else [])
else:
    df['country_list'] = [[]]

if type_col:
    df[type_col] = df[type_col].fillna('Unknown').str.title()
else:
    df['type'] = 'Unknown'
    type_col = 'type'

if rating_col:
    df[rating_col] = df[rating_col].fillna('Not Rated')
else:
    rating_col = None

# parse duration numeric
if duration_col and duration_col in df.columns:
    df['duration_num'] = df[duration_col].astype(str).str.extract(r'(\d+)').astype(float)
else:
    df['duration_num'] = np.nan

# Save a cleaned sample head for inspection
df.head(3).to_csv(os.path.join(TABLE_DIR, 'sample_head.csv'), index=False)

# -----------------------------
# 3) Analysis & Visuals
# -----------------------------
def savefig(fig, name, dpi=150):
    path = os.path.join(FIG_DIR, name)
    fig.savefig(path, bbox_inches='tight', dpi=dpi)
    print("Saved", path)

# 3.1 Movies vs TV Shows
type_counts = df[type_col].value_counts()
fig, ax = plt.subplots(figsize=(6,6))
sns.barplot(y=type_counts.index, x=type_counts.values, palette='viridis', ax=ax)
ax.set_xlabel('Count'); ax.set_ylabel('Type'); ax.set_title('Movies vs TV Shows (Counts)')
savefig(fig, 'type_counts_bar.png'); plt.close(fig)

fig, ax = plt.subplots(figsize=(6,6))
ax.pie(type_counts.values, labels=type_counts.index, autopct='%1.1f%%', startangle=140)
ax.set_title('Movies vs TV Shows (Share)')
savefig(fig, 'type_counts_pie.png'); plt.close(fig)
type_counts.to_frame('count').to_csv(os.path.join(TABLE_DIR, 'type_counts.csv'))

# 3.2 Content added over years
year_type = df.dropna(subset=['year']).groupby(['year', type_col]).size().reset_index(name='count')
if not year_type.empty:
    pivot = year_type.pivot(index='year', columns=type_col, values='count').fillna(0).sort_index()
    fig, ax = plt.subplots(figsize=(12,6))
    pivot.plot(ax=ax, marker='o'); ax.set_title('Content Added per Year by Type'); ax.set_xlabel('Year'); ax.set_ylabel('Number of Titles')
    savefig(fig, 'content_by_year_line.png'); plt.close(fig)
    fig, ax = plt.subplots(figsize=(12,6))
    pivot.plot.area(ax=ax); ax.set_title('Stacked Area: Content by Year and Type'); ax.set_xlabel('Year'); ax.set_ylabel('Number of Titles')
    savefig(fig, 'content_by_year_area.png'); plt.close(fig)
    pivot.to_csv(os.path.join(TABLE_DIR, 'type_year_pivot.csv'))

# 3.3 Top genres
genres_exploded = df[['year', 'genres_list', title_col]].explode('genres_list')
genres_exploded['genre'] = genres_exploded['genres_list'].astype(str).str.title().replace('Nan', 'Unknown')
genre_counts = genres_exploded['genre'].value_counts()
top_genres = genre_counts.head(15)
fig, ax = plt.subplots(figsize=(12,6))
sns.barplot(x=top_genres.values, y=top_genres.index, palette='coolwarm', ax=ax)
ax.set_xlabel('Number of Titles'); ax.set_ylabel('Genre'); ax.set_title('Top Genres (Overall)')
savefig(fig, 'top_genres_bar.png'); plt.close(fig)
fig, ax = plt.subplots(figsize=(7,7))
ax.pie(top_genres.head(8).values, labels=top_genres.head(8).index, autopct='%1.1f%%', startangle=140)
ax.set_title('Top 8 Genres Share'); savefig(fig, 'top_genres_pie.png'); plt.close(fig)
genre_counts.to_frame('count').to_csv(os.path.join(TABLE_DIR, 'genre_counts.csv'))

# 3.4 Genre trends over years
genre_year = genres_exploded.dropna(subset=['year']).groupby(['year', 'genre']).size().reset_index(name='count')
top5 = genre_counts.head(5).index.tolist()
fig, ax = plt.subplots(figsize=(12,6))
for g in top5:
    subset = genre_year[genre_year['genre']==g]
    ax.plot(subset['year'], subset['count'], marker='o', label=g)
ax.set_title('Top 5 Genres: Yearly Trends'); ax.set_xlabel('Year'); ax.set_ylabel('Number of Titles'); ax.legend()
savefig(fig, 'genre_trends_top5.png'); plt.close(fig)

# 3.5 Genre-year heatmap
heat = genre_year.pivot(index='genre', columns='year', values='count').fillna(0)
top25 = genre_counts.head(25).index
heat_top = heat.loc[heat.index.intersection(top25)]
fig, ax = plt.subplots(figsize=(14,8))
sns.heatmap(heat_top, cmap='YlGnBu', linewidths=.5, ax=ax)
ax.set_title('Heatmap: Genre vs Year (Top 25 Genres)'); savefig(fig, 'genre_year_heatmap.png'); plt.close(fig)
heat_top.to_csv(os.path.join(TABLE_DIR, 'genre_year_heatmap.csv'))

# 3.6 Dendrogram
if not heat_top.empty:
    from scipy.spatial.distance import pdist
    matrix = heat_top.values
    if matrix.shape[0] > 1 and np.any(np.std(matrix, axis=1) > 0):
        Z = linkage(matrix, method='ward')
        fig = plt.figure(figsize=(12, 6))
        dendrogram(Z, labels=heat_top.index, leaf_rotation=90)
        plt.title('Dendrogram: Genre similarity by Yearly Profiles')
        savefig(fig, 'genre_dendrogram.png'); plt.close(fig)

# 3.7 Country contributions
countries_exploded = df[['year', 'country_list', title_col]].explode('country_list')
countries_exploded['country'] = countries_exploded['country_list'].astype(str).str.title().replace('Nan', 'Unknown')
country_counts = countries_exploded['country'].value_counts()
top_countries = country_counts.head(15)
fig, ax = plt.subplots(figsize=(12,6))
sns.barplot(x=top_countries.values, y=top_countries.index, palette='magma', ax=ax)
ax.set_title('Top Countries by Number of Titles'); ax.set_xlabel('Number of Titles'); ax.set_ylabel('Country')
savefig(fig, 'top_countries_bar.png'); plt.close(fig)
country_counts.to_frame('count').to_csv(os.path.join(TABLE_DIR, 'country_counts.csv'))

treemap_vals = top_countries.values
treemap_labels = [f"{c}\n{v}" for c, v in zip(top_countries.index, treemap_vals)]
fig, ax = plt.subplots(figsize=(12,6))
squarify.plot(sizes=treemap_vals, label=treemap_labels, alpha=.8)
plt.axis('off'); plt.title('Treemap: Top Countries (by titles)'); savefig(fig, 'countries_treemap.png'); plt.close(fig)

# 3.8 Duration plots
fig, ax = plt.subplots(figsize=(10,6))
sns.histplot(df['duration_num'].dropna(), kde=True, bins=30, ax=ax)
ax.set_title('Distribution of Durations (numeric extracted)'); savefig(fig, 'duration_hist.png'); plt.close(fig)

if 'seasons' in df.columns or df['duration'].astype(str).str.contains('Season', case=False).any():
    tv = df[df[type_col].str.contains('tv', case=False, na=False)].copy()
    tv['seasons_num'] = tv['duration'].astype(str).str.extract(r'(\d+)').astype(float)
    fig, ax = plt.subplots(figsize=(10,5))
    sns.countplot(x=tv['seasons_num'].dropna().astype(int), order=sorted(tv['seasons_num'].dropna().unique()), ax=ax)
    ax.set_title('Count of Seasons in TV Shows'); savefig(fig, 'tv_seasons_count.png'); plt.close(fig)

# 3.9 Ratings
if rating_col:
    rating_counts = df[rating_col].value_counts().head(20)
    fig, ax = plt.subplots(figsize=(10,8))
    sns.barplot(y=rating_counts.index, x=rating_counts.values, ax=ax)
    ax.set_title('Rating Distribution (Top)'); ax.set_xlabel('Count'); ax.set_ylabel('Rating')
    savefig(fig, 'rating_distribution.png'); plt.close(fig)
    rating_counts.to_frame('count').to_csv(os.path.join(TABLE_DIR, 'rating_counts.csv'))

# -----------------------------
# 4) Summary Tables
# -----------------------------
yearly_totals = df.groupby('year').size().rename('total_titles').reset_index().dropna().sort_values('year')
yearly_totals.to_csv(os.path.join(TABLE_DIR, 'yearly_totals.csv'), index=False)

genre_year_pivot = genre_year.pivot(index='genre', columns='year', values='count').fillna(0)
genre_year_pivot.to_csv(os.path.join(TABLE_DIR, 'genre_year_pivot.csv'))

country_year = countries_exploded.groupby(['year', 'country']).size().reset_index(name='count')
country_year_pivot = country_year.pivot(index='country', columns='year', values='count').fillna(0)
country_year_pivot.to_csv(os.path.join(TABLE_DIR, 'country_year_pivot.csv'))

# -----------------------------
# 5) Automated Summary & Recommendations
# -----------------------------
total_titles = len(df)
movies_count = int((df[type_col].str.lower() == 'movie').sum())
tv_count = int((df[type_col].str.lower() == 'tv show').sum() or (df[type_col].str.lower() == 'tv').sum())
top_genre = genre_counts.idxmax() if not genre_counts.empty else 'Unknown'
top_country = country_counts.idxmax() if not country_counts.empty else 'Unknown'

summary_lines = [
    f"Report generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}",
    f"Total titles analyzed: {total_titles:,}",
    f"Movies: {movies_count:,} | TV Shows: {tv_count:,}",
    f"Top genre overall: {top_genre} ({int(genre_counts.max()) if not genre_counts.empty else 0} titles)",
    f"Top country overall: {top_country} ({int(country_counts.max()) if not country_counts.empty else 0} titles)",
    "",
    "Key Observations:",
    "- Movies vs TV Shows: see type_counts_bar.png and type_counts_pie.png for distribution.",
    "- Content growth per year: see content_by_year_line.png and content_by_year_area.png.",
    "- Genres: top_genres_bar.png and genre_trends_top5.png show most frequent genres and their trends.",
    "- Countries: top_countries_bar.png and countries_treemap.png show leading country contributors.",
    "- Heatmap: genre_year_heatmap.png shows how genres evolved year-by-year.",
    "",
    "Strategic Recommendations:",
    "1) Invest in high-growth genres identified in the last 3-5 years (refer to Top 5 Genre Trends).",
    "2) Grow regional originals in underrepresented countries that show rising trends.",
    "3) Balance investment across Movies and TV Shows based on engagement metrics (TV Shows for retention).",
    "4) Consider limited-series and documentary investments where growth is observed.",
]

summary_text = "\n".join(summary_lines)
with open(os.path.join(OUT_DIR, 'automated_summary.txt'), 'w', encoding='utf-8') as f:
    f.write(summary_text)

print("\nSummary written to", os.path.join(OUT_DIR, 'automated_summary.txt'))
print("Figures and tables saved in:", OUT_DIR)

# -----------------------------
# 6) Build PDF
# -----------------------------
pdf = FPDF(orientation='P', unit='mm', format='A4')
pdf.set_auto_page_break(auto=True, margin=12)
pdf.add_page()
pdf.set_font("Helvetica", size=16, style='B')
pdf.cell(0, 8, "Netflix Content Trends Analysis", ln=True, align='C')
pdf.ln(2)
pdf.set_font("Helvetica", size=10)
pdf.multi_cell(0, 5, "This report contains visual analysis of Netflix titles dataset: distribution by type, genre trends, country contributions, and strategic recommendations.",)
pdf.ln(3)

# Use genre_col and title_col for comparison
df['Primary_Genre'] = df[genre_col].astype(str).str.split(',').str[0].str.strip()

earliest_years = sorted(df['year'].dropna().unique())[:3]
latest_years = sorted(df['year'].dropna().unique())[-3:]

early_genres = df[df['year'].isin(earliest_years)].groupby('Primary_Genre')[title_col].count().sort_values(ascending=False)
late_genres = df[df['year'].isin(latest_years)].groupby('Primary_Genre')[title_col].count().sort_values(ascending=False)

genre_comparison = pd.DataFrame({
    f"Early ({earliest_years[0]}-{earliest_years[-1]})": early_genres,
    f"Recent ({latest_years[0]}-{latest_years[-1]})": late_genres
}).fillna(0).astype(int)

genre_comparison['Change (%)'] = ((genre_comparison.iloc[:, 1] - genre_comparison.iloc[:, 0]) / genre_comparison.iloc[:, 0].replace(0,1)) * 100

print("\n📊 Genre Share Comparison (Earliest 3 Years vs Latest 3 Years):\n")
print(genre_comparison.sort_values(by='Change (%)', ascending=False))
# 6) Build PDF (Add figures)
# -----------------------------
pdf = FPDF(orientation='P', unit='mm', format='A4')
pdf.set_auto_page_break(auto=True, margin=12)
pdf.add_page()
pdf.set_font("Helvetica", size=16, style='B')
pdf.cell(0, 8, "Netflix Content Trends Analysis", ln=True, align='C')
pdf.ln(2)
pdf.set_font("Helvetica", size=10)
pdf.multi_cell(0, 5, "This report contains visual analysis of Netflix titles dataset: distribution by type, genre trends, country contributions, and strategic recommendations.",)
pdf.ln(3)

# Add charts to PDF (example for type charts)
pdf.set_font("Helvetica", size=12, style='B')
pdf.cell(0, 6, "Movies vs TV Shows Distribution", ln=True)
pdf.image(os.path.join(FIG_DIR, 'type_counts_bar.png'), w=180)
pdf.ln(4)
pdf.image(os.path.join(FIG_DIR, 'type_counts_pie.png'), w=100)
pdf.ln(6)

# Genre charts
pdf.cell(0, 6, "Top Genres Overview", ln=True)
pdf.image(os.path.join(FIG_DIR, 'top_genres_bar.png'), w=180)
pdf.ln(4)
pdf.image(os.path.join(FIG_DIR, 'top_genres_pie.png'), w=120)
pdf.ln(6)

# (Similarly add other figures: genre trends, country treemap, heatmap, dendrogram, etc.)

pdf.output(PDF_FILE)
print("PDF report generated:", PDF_FILE)
