# importing required libraries
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import plotly.express as px
import seaborn as sns
import webbrowser
import streamlit as st
from sklearn.model_selection import train_test_split 
from sklearn.preprocessing import StandardScaler 
from sklearn.model_selection import GridSearchCV 
from sklearn.metrics import make_scorer, balanced_accuracy_score 
from sklearn import tree
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import train_test_split 
from sklearn import metrics 
from sklearn import svm
from sklearn.svm import SVC



# Load the data
nba_df = pd.read_csv(r"C:\Users\User\Downloads\Compressed\NBA dataset\NBA dataset\games_details.csv")
teams_df = pd.read_csv(r"C:\Users\User\Downloads\Compressed\NBA dataset\NBA dataset\teams.csv")
games_df = pd.read_csv(r"C:\Users\User\Downloads\Compressed\NBA dataset\NBA dataset\games.csv")
ranking_df = pd.read_csv(r"C:\Users\User\Downloads\Compressed\NBA dataset\NBA dataset\ranking.csv")


# Streamlit App Header
st.title("NBA Data Analysis")

st.header('About Project')
st.write('This Project is about NBA : I will work on analyzing how NBA teams win, and figure out which aspects during the game helped them win. The data provides player information, over 25,000 game data and statistics, team ranking since 2003, and informations of all the 30 teams in the NBA. ')
st.header('Content')
st.write('You can find 5 datasets :')

st.write('games.csv : all games from 2004 season to last update with the date, teams and some details like number of points, etc.')
st.write('games_details.csv : details of games dataset, all statistics of players for a given game')
st.write('ranking.csv : ranking of NBA given a day (split into west and east on CONFERENCE column')
st.write('teams.csv : all teams of NBA')

# Data Preprocessing

kaggle_url = 'https://www.kaggle.com/datasets/nathanlauga/nba-games'

if st.button('Go to Kaggle'):
    webbrowser.open_new_tab(kaggle_url)


# droping extra columns:'PLAYER_ID', 'NICKNAME', and 'COMMENT' in nba dataframe
nba_df.drop(['PLAYER_ID', 'NICKNAME', 'COMMENT'], axis=1, inplace=True)
# dropping extra column: 'LEAGUE_ID', 'TEAM_ID','ABBREVIATION' in teams dataframe
teams_df.drop(['TEAM_ID', 'LEAGUE_ID', 'ABBREVIATION'], axis=1, inplace=True)

# First, rename the 'CITY' column in teams_df to match the column name in nba_df
teams_df = teams_df.rename(columns={'CITY': 'TEAM_CITY'})

# Then merge the two data frames on the 'TEAM_CITY' column
df = pd.merge(nba_df, teams_df, on='TEAM_CITY', how='left')

# set the option to show all columns
pd.set_option('display.max_columns', None)


# get the current column order
current_cols = df.columns.tolist()

# define the new column order with 'NICKNAME' next to 'TEAM_CITY'
new_cols = ['GAME_ID', 'TEAM_ABBREVIATION', 'TEAM_CITY', 'NICKNAME', 'PLAYER_NAME', 'START_POSITION', 'MIN', 'FGM', 'FGA', 'FG_PCT',
            'FG3M', 'FG3A', 'FG3_PCT', 'FTM', 'FTA', 'FT_PCT', 'OREB', 'DREB', 'REB', 'AST', 'STL', 'BLK', 'TO', 'PF',
            'PTS', 'PLUS_MINUS', 'MIN_YEAR', 'MAX_YEAR', 'YEARFOUNDED', 'ARENA', 'ARENACAPACITY',
            'OWNER', 'GENERALMANAGER', 'HEADCOACH', 'DLEAGUEAFFILIATION']

# update the column order
df = df.reindex(columns=new_cols)
st.write('All Columns: ')
# all columns of datafrane after merge two data frame
df.columns

#aggregate of duplicated rows
df.duplicated().sum()

#dropping all duplicated rows
df.drop_duplicates(inplace=True)

#ropping all extra teams that they are not in franchise now
df = df[df['TEAM_CITY'] != 'New Orleans/Oklahoma City']
df = df[df['TEAM_CITY'] != 'Seattle']
df = df[df['TEAM_CITY'] != 'New Jersey']



st.write('Null value of DATA SET AS Boolean shows :')
st.write(df.isnull())

st.write('Sum of all Null value : ')
st.write(df.isna().sum().to_frame().T.style.set_properties(**{"background-color": "lightblue","color":"#452912","border": "1.5px #ddab46"})
)


st.write('Check for null values:')
for col in df.columns:
    null_rate = df[col].isnull().sum() / len(df) * 100
    if null_rate > 0:
        st.write(f'Percentage of null values in {col}: {null_rate:.2f}%')



st.markdown('**Ploting heatmap of null value : **')

fig=plt.figure(figsize=(8,4))
sns.heatmap(df.isnull(),cmap='cividis', cbar_kws={'label': 'Missing Values'})
st.pyplot(fig)




#showing all nan value
df.isna().sum()

#filling nan value by mean and mode technique
df['PLUS_MINUS'].fillna(df['PLUS_MINUS'].mean(), inplace=True)
df['START_POSITION'].fillna(df['START_POSITION'].mode()[0], inplace=True)
df['MIN'] = pd.to_numeric(df['MIN'], errors='coerce')  # Convert 'MIN' column to numeric

df['MIN'].fillna(df['MIN'].mean(), inplace=True)

#dropping redundent colmun
df.dropna(subset=['NICKNAME'], inplace=True)

#filling nan value by mean technique
df['FGM'].fillna(df['FGM'].mean(), inplace=True)
df['FGA'].fillna(df['FGA'].mean(), inplace=True)
df['FG_PCT'].fillna(df['FG_PCT'].mean(), inplace=True)
df['FG3M'].fillna(df['FG3M'].mean(), inplace=True)
#filling nan value by mean and mode technique
df['FG3A'].fillna(df['FG3A'].mean(), inplace=True)
df['FG3_PCT'].fillna(df['FG3_PCT'].mean(), inplace=True)
df['FTM'].fillna(df['FTM'].mean(), inplace=True)
df['FTA'].fillna(df['FTA'].mean(), inplace=True)
df['FT_PCT'].fillna(df['FT_PCT'].mean(), inplace=True)
df['OREB'].fillna(df['OREB'].mean(), inplace=True)
df['DREB'].fillna(df['DREB'].mean(), inplace=True)
df['REB'].fillna(df['REB'].mean(), inplace=True)
df['AST'].fillna(df['AST'].mean(), inplace=True)
df['STL'].fillna(df['STL'].mean(), inplace=True)
df['BLK'].fillna(df['BLK'].mean(), inplace=True)
df['TO'].fillna(df['TO'].mean(), inplace=True)
df['PF'].fillna(df['PF'].mean(), inplace=True)
df['PTS'].fillna(df['PTS'].mean(), inplace=True)
df['ARENACAPACITY'].fillna(df['ARENACAPACITY'].mean(), inplace=True)

# the count of empty values
df.isna().sum()

st.write('Heatmap of Nan values after data cleaning :')
#heatmap of nan values after data cleaning
sns.heatmap(df.isnull(), cmap='viridis')
plt.title('Missing Values Heatmap')
plt.show()


st.subheader('Data Exploration and Cleaning')
st.write('Glimpse of Dataset :')
st.write(df.head())

st.write('Dimensions of Datasets are :')
st.write(df.shape)

st.write('Summary of Dataset :')
st.write(df.head())

st.write('Information about dataset : ')
st.write(df.info())

st.write('types of each features : ')
st.write(df.dtypes)

st.write('Correlation between features : ')
st.write(df.corr())

# Print the desired information
st.write('Total number of teams:', df['TEAM_CITY'].nunique())
st.write('Total number of players:', df['PLAYER_NAME'].nunique())
st.write('Total number of types of start positions:', df['START_POSITION'].nunique())

st.write('Number of unique values in each column:')
st.write(df.nunique())

st.subheader('Data Visualization')


st.write('Plot the count of occurrences by start position and Distribution of Value : ')

# Set the style
sns.set_style('whitegrid')

# Create the figure and subplots
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 8))

# Plot the count of occurrences by start position
sns.countplot(data=df, x='START_POSITION', palette='viridis', ax=ax1)
ax1.set_title("Occurrences by Start Position")
ax1.set_xlabel("Start Position")
ax1.set_ylabel("PTS")

# Plot the distribution of values by start position
sns.boxplot(data=df, x='START_POSITION', y='PTS', palette='magma', ax=ax2)
ax2.set_title("Distribution of Values by Start Position")
ax2.set_xlabel("Start Position")
ax2.set_ylabel("Value")

# Adjust the spacing between subplots
fig.subplots_adjust(wspace=0.4)

# Display the plot in Streamlit
st.pyplot(fig)

st.write('displot with each start position : ')

# Create the displot with columns for each start position
displot = sns.displot(data=df, x="PTS", col="START_POSITION", binwidth=3, height=3, facet_kws=dict(margin_titles=True))

# Display the plot in Streamlit
st.pyplot(displot)

st.write('Plot the Distribution of Start Positions : ')

# Set up the data
st.write('Plot the Distribution of Start Positions : ')
labels = ["Guard", "Forward", "Center"]
sizes = [df["START_POSITION"].value_counts().values[0], df["START_POSITION"].value_counts().values[1], df["START_POSITION"].value_counts().values[2]]
colors = ["lightgrey", "lightcoral", "lightblue"]

# Create the pie chart
fig, ax = plt.subplots(figsize=(6, 6))
ax.pie(sizes, labels=labels, autopct="%.2f%%", startangle=8, colors=colors,
       wedgeprops=dict(width=0.5, edgecolor="black", linewidth=2), textprops={"fontsize": 16})
ax.set_title("Distribution of Start Positions", fontsize=16)

# Display the plot in Streamlit
st.pyplot(fig)


st.write('Scatter plot : ')
# Create the scatter plot
fig = px.scatter(df, x="PTS", y="FG3M", hover_data=['BLK'], color="PTS",
                 title='Scatter Plot of Points and Three-Point Field Goals Made',
                 color_continuous_scale='Viridis')

# Customize the layout
fig.update_layout(
    xaxis_title='Points (PTS)',
    yaxis_title='Three-Point Field Goals Made (FG3M)',
    hoverlabel=dict(bgcolor='white', font_size=12),
    font=dict(size=14),
    template='plotly_white'
)

# Display the plot in Streamlit
st.plotly_chart(fig)




# Create the scatter plot
fig = px.scatter(df, x='PTS', y='FTM', size='FTM', color='FTM', 
                 hover_data=[df.index], color_continuous_scale='Blues')
fig.update_traces(marker=dict(line=dict(width=1)))

# Display the plot in Streamlit
st.plotly_chart(fig)


LD77_df =df[df['PLAYER_NAME'] == "Luka Doncic"]
LD77_df.head()
LD77_df.loc[(LD77_df.GAME_ID >= 22000079), 'SEASON'] = '20-21'

def category_season(x):
    if x>=22000079:
        return '20-21'
    
    elif x>=21900119:
        return '19-20'
    
    elif x>=21800197:
        return '18-19'
    
    elif x>=21700013:
        return '17-18'
    
    
    LD77_df['SEASON'] = LD77_df.GAME_ID.apply(category_season)

LD77_df_new = LD77_df[['PLAYER_NAME', 'PTS', 'AST', 'REB', 'FG_PCT', 'FG3_PCT', 'SEASON']]

# Calculate the mean points per season
mean_pts_per_season = LD77_df_new.groupby('SEASON')['PTS'].mean()

# Create the bar plot
plt.figure(figsize=(10, 6))
mean_pts_per_season.plot(kind='bar')
plt.xlabel('Season')
plt.ylabel('Mean Points')
plt.title('Mean Points per Season for Luka Doncic')

# Display the plot in Streamlit
st.pyplot()


# Calculate the mean metrics per season
metrics_per_season = LD77_df_new.groupby('SEASON')['PTS', 'REB', 'AST'].mean()

# Create the bar plot
plt.figure(figsize=(10, 6))
metrics_per_season.plot(kind='bar')
plt.xlabel('Season')
plt.ylabel('Mean Value')
plt.title('Mean Metrics per Season for Luka Doncic')
plt.legend(['PTS', 'REB', 'AST', 'BLK'])

# Display the plot in Streamlit
st.pyplot()



# Calculate the mean metrics per season
metrics_per_season = LD77_df_new.groupby('SEASON')['PTS', 'REB', 'AST'].mean()

# Create the bar plot
plt.figure(figsize=(10, 6))
metrics_per_season.plot(kind='bar')
plt.xlabel('Season')
plt.ylabel('Mean Value')
plt.title('Mean Metrics per Season for Luka Doncic')
plt.legend(['PTS', 'REB', 'AST', 'BLK'])

# Display the plot in Streamlit
st.pyplot()



# Create the scatter plot
fig = px.scatter(LD77_df_new, x="SEASON", y="PTS", size="PTS", color="PTS", hover_data=['FG3_PCT'],
                 title="Scatter Plot of PTS by Season",
                 labels={'SEASON': 'Season', 'PTS': 'PTS', 'FG3_PCT': 'FG3 Percentage'})

# Display the plot in Streamlit
st.plotly_chart(fig)



# Create the grouped bar plot
fig = px.bar(LD77_df_new, x="SEASON", y=["PTS", "AST", "REB"], barmode="group",
             labels={'SEASON': 'Season', 'variable': 'Metric', 'value': 'Value'},
             title="Grouped Bar Plot of PTS, AST, and REB by Season")

# Display the plot in Streamlit
st.plotly_chart(fig)



# Create the scatter matrix plot
fig = px.scatter_matrix(LD77_df_new, dimensions=["SEASON", "PTS", "AST", "REB"], color="SEASON",
                        labels={'SEASON': 'Season', 'PTS': 'PTS', 'AST': 'AST', 'REB': 'REB'})

# Display the plot in Streamlit
st.plotly_chart(fig)


# Create the scatter ternary plot
fig = px.scatter_ternary(LD77_df_new, a="PTS", b="AST", c="REB", hover_name="SEASON",
                         color="SEASON", size="PTS", size_max=15,
                         labels={'PTS': 'PTS', 'AST': 'AST', 'REB': 'REB', 'SEASON': 'Season'})

# Display the plot in Streamlit
st.plotly_chart(fig)


# Create the correlation heatmap
f, ax = plt.subplots(figsize=(12, 8))
sns.heatmap(np.round(df.corr(), 2), annot=True)

# Display the heatmap in Streamlit
st.pyplot(f)


# Plot the correlation between points and 3-pointers scored by position
plt.figure(figsize=(10, 6))
sns.scatterplot(x="FG3M", y="PTS", hue='START_POSITION', data=df, palette="Set1")
sns.regplot(x="FG3M", y="PTS", data=df, scatter=False, color='black', ci=None)
plt.title("Correlation between 3-Pointers Made and Points by Position", fontsize=16)
plt.xlabel("3-Point Field Goals Made (FG3M)")
plt.ylabel("Points (PTS)")
plt.legend(title="Position", loc="upper right")

# Display the plot in Streamlit
st.pyplot(plt)



player = 'PLAYER_NAME'

DL = 'Damian Lillard'
RW = 'Russell Westbrook'
JT = 'Jayson Tatum'
JE = 'Joel Embiid'

player_list = (df[player] == DL) | (df[player] == RW) | (df[player] == JT) | (df[player] == JE)
game_stat = [player, 'PTS', 'AST', 'REB']




df[player_list].head(10)

df[game_stat].head(10)

df[player_list][game_stat].head(10)

df_player_list = df[player_list]

df_player_list[game_stat]

# Create the cross-tabulation table
cross_tab = pd.crosstab(df_player_list[player], df_player_list['PTS'], margins=True).style.background_gradient(cmap='viridis')

# Display the cross-tabulation table in Streamlit
st.dataframe(cross_tab)



# Create the kernel density plot
fig, ax = plt.subplots(1, 1, figsize=(20, 8))
sns.kdeplot(df[df[player] == DL]['PTS'], ax=ax)
sns.kdeplot(df[df[player] == JT]['PTS'], ax=ax)
plt.legend(['Damian Lillard', 'Jayson Tatum'])
plt.title("Kernel Density Plot of Points Scored by Players")
plt.xlabel("Points (PTS)")
plt.ylabel("Density")

# Display the plot in Streamlit
st.pyplot(fig)


# Create the scatter plot
fig = px.scatter(df_player_list, x=player, y='PTS', color='TEAM_ABBREVIATION',
                 size='PTS', hover_name='PLAYER_NAME')

# Update the layout
fig.update_layout(
    title='Player Points (PTS) Scatter Plot',
    xaxis_title='Player',
    yaxis_title='Points (PTS)',
    legend_title='Team',
    hoverlabel=dict(bgcolor='white', font_size=12),
    font=dict(size=14),
    template='plotly_white'
)

# Display the plot in Streamlit
st.plotly_chart(fig)



# Create the scatter plot
fig = px.scatter(df_player_list, x=player, y='REB', color='TEAM_ABBREVIATION',
                 size='REB', hover_name='PLAYER_NAME')

# Update the layout
fig.update_layout(
    title='Player Rebounds (REB) Scatter Plot',
    xaxis_title='Player',
    yaxis_title='Rebounds (REB)',
    legend_title='Team',
    hoverlabel=dict(bgcolor='white', font_size=12),
    font=dict(size=14),
    template='plotly_white'
)

# Display the plot in Streamlit
st.plotly_chart(fig)




# Create the scatter plot
fig = px.scatter(df_player_list, x=player, y='AST', color='TEAM_ABBREVIATION',
                 size='AST', hover_name='PLAYER_NAME')

# Update the layout
fig.update_layout(
    title='Player Assists (AST) Scatter Plot',
    xaxis_title='Player',
    yaxis_title='Assists (AST)',
    legend_title='Team',
    hoverlabel=dict(bgcolor='white', font_size=12),
    font=dict(size=14),
    template='plotly_white'
)

# Display the plot in Streamlit
st.plotly_chart(fig)



# Create the subplots
f, ax = plt.subplots(1, 3, figsize=(20, 5))

# Plot the violin plots
sns.violinplot(x=player, y='PTS', hue='TEAM_ABBREVIATION', data=df_player_list, scale='count', split=False, ax=ax[0])
ax[0].set_title('PTS and Player for Team')
ax[0].set_yticks(range(0, 80, 10))

sns.violinplot(x=player, y='AST', hue='TEAM_ABBREVIATION', data=df_player_list, scale='count', split=False, ax=ax[1])
ax[1].set_title('AST and Player for Team')
ax[1].set_yticks(range(0, 25, 5))

sns.violinplot(x=player, y='REB', hue='TEAM_ABBREVIATION', data=df_player_list, scale='count', split=False, ax=ax[2])
ax[2].set_title('REB and Player for Team')
ax[2].set_yticks(range(0, 25, 5))

# Adjust the layout and display the plot in Streamlit
plt.tight_layout()
st.pyplot(f)


# Set the figure size
plt.figure(figsize=(18, 5))

# Plot the bar plots
plt.subplot(131)
sns.barplot(x=player, y='PTS', hue='TEAM_ABBREVIATION', data=df_player_list, estimator=sum, ci=None)
plt.title('Total Points (PTS) by Player and Team')
plt.ylabel('Total Points')

plt.subplot(132)
sns.barplot(x=player, y='AST', hue='TEAM_ABBREVIATION', data=df_player_list, estimator=sum, ci=None)
plt.title('Total Assists (AST) by Player and Team')
plt.ylabel('Total Assists')

plt.subplot(133)
sns.barplot(x=player, y='REB', hue='TEAM_ABBREVIATION', data=df_player_list, estimator=sum, ci=None)
plt.title('Total Rebounds (REB) by Player and Team')
plt.ylabel('Total Rebounds')

plt.tight_layout()

# Display the plot in Streamlit
st.pyplot(plt)


df_team_ranking = ranking_df[(ranking_df['TEAM'] == 'Boston') | (ranking_df['TEAM'] == 'Philadelphia')]


# Set the figure size
plt.figure(figsize=(12, 8))

# Plot the bar plot
ranking_df.groupby('TEAM')['W_PCT'].mean().sort_values().plot(kind='barh', color='skyblue')

plt.title('Average Win Percentage by Team')
plt.xlabel('Win Percentage')
plt.ylabel('Team')

plt.xticks(rotation=45)
plt.gca().invert_yaxis()

# Display the plot in Streamlit
st.pyplot(plt)



# Select columns from the DataFrame
game_stat = ['PTS', 'FGA']
df_game_stat = df[game_stat]

# Set the plot style and size
sns.set(style='darkgrid')
plt.figure(figsize=(12, 8))

# Create the categorical plot (point plot)
sns.catplot(x='FGA', y='PTS', hue='PLAYER_NAME', data=df_player_list, kind='point', height=10, aspect=2)

# Set the plot title and axis labels
plt.title('Game Statistics: Points (PTS) vs. Field Goal Attempts (FGA)')
plt.xlabel('Field Goal Attempts (FGA)')
plt.ylabel('Points (PTS)')

# Display the plot in Streamlit
st.pyplot(plt)



# Calculate the percentage of home team wins and losses
pct_home_win = games_df['HOME_TEAM_WINS'].value_counts() / len(games_df) * 100

# Display the results
st.write(f'Teams are likely to win {pct_home_win[1]:.2f}% during home games, and lose {pct_home_win[0]:.2f}% during home games')




# Calculate the number of wins for home games and all games
win_filt = games_df[games_df['HOME_TEAM_WINS'] == True]
lose_filt = games_df[games_df['HOME_TEAM_WINS'] == False]

x = win_filt['HOME_TEAM_WINS'].value_counts()
y = lose_filt['HOME_TEAM_WINS'].value_counts()

ti = [0.5]
hor = np.arange(len(ti))

plt.bar(ti, x, width=0.25, color='#0077b6', label='Home Games')
plt.bar(hor + 0.75, y, width=0.25, color='#fb8500', label='Away Games')

plt.ylabel('Number of Wins')
plt.xticks(color='w')
plt.title('Win comparison between Home and Away Games')
plt.legend()

st.pyplot()



def get_mean(group, column):
    return group[column].mean()

x = [
    get_mean(games_df, 'PTS_home'),
    get_mean(win_filt, 'AST_home'),
    get_mean(games_df, 'REB_home')
]
y = [
    get_mean(games_df, 'PTS_away'),
    get_mean(win_filt, 'AST_away'),
    get_mean(games_df, 'REB_away')
]

ti = ['Points Allowed', 'Assist', 'Rebound']
hor = range(len(ti))

plt.bar(hor, x, width=0.25, color='#0077b6', label='Home Win Games')
plt.bar([val + 0.25 for val in hor], y, width=0.25, color='#fb8500', label='Away Lose Games')
plt.xlabel('Metrics')
plt.ylabel('Mean Value')
plt.title('Comparison of Mean Metrics: Home Win Games vs Away Lose Games')
plt.xticks(hor, ti)
plt.legend()

st.pyplot()



# Grouping by rebounds and home team wins
reb_grp = games_df.groupby(['REB_home', 'HOME_TEAM_WINS'])
reb_table = reb_grp.size().unstack(fill_value=0)

# Plotting rebounds of home winners and losers
fig, ax = plt.subplots(1, 2, figsize=(18.5, 6))

ax[0].plot(reb_table[1], color='#33AB5F', label='Rebounds of Home Winner')
ax[0].plot(reb_table[0], color='#BA0001', label='Rebounds of Home Loser')
ax[0].set_ylabel('Number of Games')
ax[0].set_xlabel('Rebounds per Game')
ax[0].legend()

# Grouping by rebounds and home team wins for away games
losereb_grp = games_df.groupby(['REB_away', 'HOME_TEAM_WINS'])
losereb_table = losereb_grp.size().unstack(fill_value=0)

# Plotting rebounds of away winners and losers
ax[1].plot(losereb_table[0], color='#33AB5F', label='Rebounds of Away Winner')
ax[1].plot(losereb_table[1], color='#BA0001', label='Rebounds of Away Loser')
ax[1].set_xticks([])
ax[1].set_yticks([])
ax[1].set_xlim([15, 75])
ax[1].set_xlabel('Rebounds per Game')
ax[1].legend()

st.pyplot(fig)




# Grouping and making the table for field goal percentage
fgpct_grp = games_df.groupby(['FG_PCT_home', 'HOME_TEAM_WINS'])
fgpct_table = fgpct_grp.size().unstack(fill_value=0)
fgpct_grp_lose = games_df.groupby(['FG_PCT_away', 'HOME_TEAM_WINS'])
fgpct_away_table = fgpct_grp_lose.size().unstack(fill_value=0)

# Grouping and making the table for 3-point field goal percentage
fg3pct_grp = games_df.groupby(['FG3_PCT_home', 'HOME_TEAM_WINS'])
fg3pct_table = fg3pct_grp.size().unstack(fill_value=0)
fg3pct_grp_lose = games_df.groupby(['FG3_PCT_away', 'HOME_TEAM_WINS'])
fg3pct_away_table = fg3pct_grp_lose.size().unstack(fill_value=0)

plt.subplot(1, 2, 1)
plt.plot(fgpct_table[1], color='#33AB5F', label='Field Goal (%) of Home Winner')
plt.plot(fgpct_table[0], color='#BA0001', label='Field Goal (%) of Home Loser')
plt.xlabel('Field Goal (%) per Game')
plt.legend(loc='upper right')

plt.subplot(1, 2, 2)
plt.plot(fg3pct_away_table[0], color='#00FF00', label='3 Pt Field Goal (%) of Away Winner')
plt.plot(fg3pct_away_table[1], color='#BA0001', label='3 Pt Field Goal (%) of Away Loser')
plt.xlabel('Field Goal (%) per Game')
plt.legend(loc='upper right')

fig = plt.gcf()
fig.set_size_inches(18.5, 6)

st.pyplot()






#Predictive Models

# sort dataframe by date
games_df = games_df.sort_values(by='GAME_DATE_EST').reset_index(drop = True)
# drop empty entries, data before 2004 contains NaN
games_df = games_df.loc[games_df['GAME_DATE_EST'] >= "2004-01-01"].reset_index(drop=True)
# check null
games_df.isnull().values.any() 


selected_features = [
    'FG_PCT_home', 'FT_PCT_home', 'FG3_PCT_home', 'AST_home', 'REB_home',
    'FG_PCT_away', 'FT_PCT_away', 'FG3_PCT_away', 'AST_away', 'REB_away',
    ]

# check the features we selected
X = games_df[selected_features]
X.head()

y = games_df['HOME_TEAM_WINS']
y.head()

X = X.to_numpy()
y = y.to_numpy()


X_train, X_test, y_train, y_test = train_test_split(
    X, 
    y, 
    test_size=0.3, random_state=42)

print("X shape", X_train.shape, "y shape", y_train.shape)




#RandomForest
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report


# Create and fit the RandomForestClassifier
random = RandomForestClassifier(n_estimators=100, criterion='entropy')
random.fit(X_train, y_train)
y_predict = random.predict(X_test)

# Calculate the classification report and cross-tabulation
report = classification_report(y_test, y_predict)
cross_tab = pd.crosstab(y_test, y_predict)

# Set the Streamlit app title
st.title("RandomForest")

# Display the classification report
st.write("Classification Report:")
st.write(report)

# Display the cross-tabulation
st.write("Cross-Tabulation:")
st.write(cross_tab)



import streamlit as st
from xgboost import XGBClassifier
from sklearn.metrics import classification_report, confusion_matrix
import seaborn as sns


#XGBoost
# Create and fit the XGBoost classifier
model = XGBClassifier()
model.fit(X_train, y_train)
y_predict = model.predict(X_test)

# Calculate the classification report and confusion matrix
report = classification_report(y_test, y_predict)
cm = confusion_matrix(y_test, y_predict)

# Set the Streamlit app title
st.title("XGBoost Classifier")

# Display the classification report
st.write("Classification Report:")
st.write(report)

# Display the confusion matrix as a heatmap
st.write("Confusion Matrix:")
sns.heatmap(cm, annot=True)
st.pyplot()


import streamlit as st
from catboost import CatBoostClassifier
from sklearn.metrics import classification_report, confusion_matrix
import seaborn as sns


#CatBoost

# Create and fit the CatBoost classifier
cat = CatBoostClassifier()
cat.fit(X_train, y_train)
y_predict = cat.predict(X_test)

# Calculate the classification report and confusion matrix
report = classification_report(y_test, y_predict)
cm = confusion_matrix(y_test, y_predict)

# Set the Streamlit app title
st.title("CatBoost Classifier")

# Display the classification report
st.write("Classification Report:")
st.write(report)

# Display the confusion matrix as a heatmap
st.write("Confusion Matrix:")
sns.heatmap(cm, annot=True)
st.pyplot()


import streamlit as st
from sklearn.naive_bayes import GaussianNB
from sklearn.metrics import classification_report, confusion_matrix
import seaborn as sns
import pandas as pd

#Gaussian Naive Bayes

# Create and fit the Gaussian Naive Bayes classifier
model = GaussianNB()
model.fit(X_train, y_train)
y_predict = model.predict(X_test)

# Calculate the classification report and confusion matrix
report = classification_report(y_test, y_predict)
cm = confusion_matrix(y_test, y_predict)

# Calculate the cross-tabulation
cross_tab = pd.crosstab(y_test, y_predict)

# Set the Streamlit app title
st.title("Gaussian Naive Bayes Classifier")

# Display the classification report
st.write("Classification Report:")
st.write(report)

# Display the confusion matrix as a heatmap
st.write("Confusion Matrix:")
sns.heatmap(cm, annot=True)
st.pyplot()

# Display the cross-tabulation
st.write("Cross-Tabulation:")
st.write(cross_tab)



import streamlit as st
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report
import pandas as pd


#Logistic Regression

# Create and fit the Logistic Regression classifier
lo = LogisticRegression(random_state=0)
lo.fit(X_train, y_train)
y_predict = lo.predict(X_test)

# Calculate the classification report
report = classification_report(y_test, y_predict)

# Calculate the cross-tabulation
cross_tab = pd.crosstab(y_test, y_predict)

# Set the Streamlit app title
st.title("Logistic Regression Classifier")

# Display the classification report
st.write("Classification Report:")
st.write(report)

# Display the cross-tabulation
st.write("Cross-Tabulation:")
st.write(cross_tab)




import streamlit as st
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import classification_report
import pandas as pd


#KNN
# Create and fit the KNN classifier
classifier = KNeighborsClassifier(n_neighbors=5, metric='minkowski', p=2)
classifier.fit(X_train, y_train)
y_predict = classifier.predict(X_test)

# Calculate the classification report
report = classification_report(y_test, y_predict)

# Calculate the cross-tabulation
cross_tab = pd.crosstab(y_test, y_predict)

# Set the Streamlit app title
st.title("K-Nearest Neighbors (KNN) Classifier")

# Display the classification report
st.write("Classification Report:")
st.write(report)

# Display the cross-tabulation
st.write("Cross-Tabulation:")
st.write(cross_tab)



scaler = StandardScaler() # initialize an instance 
X_train = scaler.fit_transform(X_train) 

# train SVM

clf = svm.SVC(kernel='linear') # initialize a model
clf.fit(X_train, y_train) # fit(train) it with the training data and targets

# check test score 
y_pred = clf.predict(X_test) 
print('balanced accuracy score:', balanced_accuracy_score(y_test, y_pred)) 



from sklearn.model_selection import GridSearchCV
from sklearn.metrics import make_scorer, balanced_accuracy_score
from sklearn import svm
from sklearn.tree import DecisionTreeClassifier
from sklearn import metrics



# Define the scoring metric
scoring = make_scorer(balanced_accuracy_score)

# Define the parameter grid for the SVM classifier
param_grid = {'C': [0.1, 1, 10],  
              'gamma': [1, 0.1, 0.01]}

# Create the GridSearchCV object
grid = GridSearchCV(svm.SVC(kernel='linear'), param_grid, scoring=scoring, refit=True, verbose=2)

# Fit the grid search to the training data
grid.fit(X_train, y_train)

# Get the best estimator from the grid search
best_estimator = grid.best_estimator_


#Decision Tree

# Create and fit the Decision Tree classifier
clf = DecisionTreeClassifier(criterion='entropy', splitter='best', random_state=1)
clf.fit(X_train, y_train)

# Predict on the test set
y_pred = clf.predict(X_test)

# Calculate and print the accuracy score
print("Accuracy:", metrics.accuracy_score(y_test, y_pred))











