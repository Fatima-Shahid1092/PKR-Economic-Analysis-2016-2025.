mport pandas as pd
import matplotlib.pyplot as plt

df = pd.read_csv('/content/infaltion new.csv')

# 2. Column Names
df.columns = df.columns.str.strip()
df['Annual Inflation(%)'] = df['Annual Inflation(%)'].astype(str).str.replace('%', '', regex=False)
df['Annual Inflation(%)'] = pd.to_numeric(df['Annual Inflation(%)'], errors='coerce')

# 4. Handling Dates & Years
df['Month-Year'] = pd.to_datetime(df['Month-Year'], errors='coerce')
df['Year'] = df['Month-Year'].dt.year

# 5. Calculating Yearly Averages (2017-2025)
yearly_avg = df.dropna(subset=['Annual Inflation(%)', 'Year'])
data_2017_2025 = yearly_avg[yearly_avg['Year'] >= 2017].groupby('Year')['Annual Inflation(%)'].mean().reset_index()

# 6. Manual 2016 Benchmark (3.77%)
manual_2016 = pd.DataFrame({'Year': [2016], 'Annual Inflation(%)': [3.77]})
final_data = pd.concat([manual_2016, data_2017_2025]).sort_values('Year')
final_data = final_data[final_data['Year'] <= 2025]

# 7. Line Graph
plt.figure(figsize=(10, 6), dpi=300)
plt.plot(final_data['Year'], final_data['Annual Inflation(%)'],
         marker='o', markersize=8, linestyle='-', color='#c0392b', linewidth=2.5, label='Avg. Inflation %')

# Adding Labels 
for i, txt in enumerate(final_data['Annual Inflation(%)']):
    plt.annotate(f"{txt:.1f}%", (final_data['Year'].iloc[i], final_data['Annual Inflation(%)'].iloc[i]),
                 textcoords="offset points", xytext=(0,10), ha='center', fontweight='bold', color='#2c3e50')

# Formatting
plt.title('The Escalating Inflation Crisis (2016–2025)', fontsize=15, fontweight='bold', pad=20)
plt.xlabel('Year(2016-2025)', fontsize=12)
plt.ylabel('Average Annual Inflation (%)', fontsize=12)
plt.xticks(final_data['Year'].astype(int), fontweight='bold')
plt.grid(True, linestyle='--', alpha=0.5)
plt.gca().spines[['top', 'right']].set_visible(False)

plt.tight_layout()
plt.show()

#--------------SECOND GRAPH---------
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from matplotlib.collections import LineCollection
import numpy as np

file_path = '/content/infaltion new.csv'
df = pd.read_csv(file_path)
df.columns = df.columns.str.strip().str.lower()
df['month-year'] = pd.to_datetime(df['month-year'], errors='coerce')

val_col = 'purchasing power of 1,000 pkr'
if val_col in df.columns:
    df[val_col] = pd.to_numeric(df[val_col].astype(str).str.replace(',', ''), errors='coerce')

comm_col = [c for c in df.columns if 'comment' in c][0]
df = df.dropna(subset=['month-year', val_col]).sort_values('month-year')

x = mdates.date2num(df['month-year'])
y = df[val_col].values
points = np.array([x, y]).T.reshape(-1, 1, 2)
segments = np.concatenate([points[:-1], points[1:]], axis=1)

colors = []
for val in y[:-1]:
    if val >= 900: colors.append('#1B5E20')    # Deep Forest Green
    elif 400 <= val < 900: colors.append('#E65100') # Rich Burnt Orange
    else: colors.append('#B71C1C')           # Deep Crimson

lc = LineCollection(segments, colors=colors, linewidth=5, zorder=2, antialiaseds=True)

plt.rcParams['font.family'] = 'serif'
plt.rcParams['text.antialiased'] = True # Ensures text is perfectly smooth
fig, ax = plt.subplots(figsize=(20, 11), dpi=300)
ax.add_collection(lc)
ax.xaxis_date()
ax.autoscale_view()
ax.set_ylim(300, 1050)
ax.set_facecolor('white')

for i, row in df.iterrows():
    comment_text = str(row[comm_col]).strip()
    if comment_text != "" and comment_text.lower() != 'nan':
        ax.scatter(row['month-year'], row[val_col], color='black', s=180, zorder=5)
        ax.annotate(comment_text.upper(),
                     xy=(row['month-year'], row[val_col]),
                     xytext=(25, 45), textcoords='offset points',
                     fontsize=16, fontweight='bold', color='#000000',
                     arrowprops=dict(arrowstyle='-|>', color='black', lw=2, connectionstyle="angle,angleA=0,angleB=90,rad=10"),
                     zorder=6)

plt.title('PURCHASING POWER EROSION OF 1,000 PKR (2016-2025)',
          fontsize=28, fontweight='bold', pad=50, color='#000000') # Black & sized down
plt.ylabel('PURCHASING POWER (PKR)', fontsize=24, fontweight='bold', labelpad=35, color='#000000')
plt.xlabel('FISCAL TIMELINE (2016 - 2025)', fontsize=24, fontweight='bold', labelpad=35, color='#000000')

plt.xticks(fontsize=20, fontweight='bold', color='#000000')
plt.yticks(fontsize=20, fontweight='bold', color='#000000')
plt.grid(True, axis='y', linestyle=':', alpha=0.3, color='black')

legend_elements = [
    plt.Line2D([0], [0], color='#1B5E20', lw=10, label='STABILITY ZONE (> 900)'),
    plt.Line2D([0], [0], color='#E65100', lw=10, label='DEVALUATION PHASE (900 - 400)'),
    plt.Line2D([0], [0], color='#B71C1C', lw=10, label='CRITICAL COLLAPSE (< 400)')
]
ax.legend(handles=legend_elements, loc='upper right', prop={'size': 20, 'weight': 'bold'},
          frameon=True, shadow=False, borderpad=1.5, edgecolor='black')

for spine in ax.spines.values():
    spine.set_linewidth(2)

plt.tight_layout()
plt.show()

#------------THIRD GRAPH---------
import matplotlib.pyplot as plt
import pandas as pd
from google.colab import files

df['Year'] = df['Date'].dt.year
yearly_stats = df.groupby('Year')['Purchasing Power of 1,000 PKR'].agg(['max', 'min'])

plt.style.use('seaborn-v0_8-white')
fig, ax = plt.subplots(figsize=(16, 9), facecolor='#1e293b') # Larger canvas
ax.set_facecolor('#1e293b')

ax.bar(yearly_stats.index, yearly_stats['max'], color='#22c55e', alpha=0.3, label='Highest Value (Strength)')
ax.bar(yearly_stats.index, yearly_stats['min'], color='#ef4444', alpha=0.8, label='Lowest Value (Devaluation)')

for year in yearly_stats.index:
    high = yearly_stats.loc[year, 'max']
    low = yearly_stats.loc[year, 'min']

    ax.text(year, high + 15, f'{int(high)}', ha='center', color='#86efac', fontweight='bold', fontsize=16)
    # Label the Devaluation (Min) - Positioned lower for clarity
    ax.text(year, low - 45, f'{int(low)}', ha='center', color='#fca5a5', fontweight='bold', fontsize=16)

plt.title('ANNUAL STRENGTH vs. DEVALUATION (Exact PKR Values)', color='white', fontsize=28, fontweight='bold', pad=35)
plt.ylabel('Value of 1,000 PKR', color='#94a3b8', fontsize=20, fontweight='bold', labelpad=20)
plt.xlabel('Timeline: Years (2016-2025)', color='#94a3b8', fontsize=20, fontweight='bold', labelpad=20)

plt.xticks(yearly_stats.index, color='#cbd5e1', fontweight='bold', fontsize=18)
plt.yticks(color='#cbd5e1', fontweight='bold', fontsize=18)

plt.grid(axis='y', linestyle='--', alpha=0.1)
ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)

plt.legend(frameon=False, labelcolor='white', loc='upper right', fontsize=16)

plt.tight_layout()

plt.show()
