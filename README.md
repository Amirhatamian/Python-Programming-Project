# NBA Data Analysis Dashboard with Streamlit

An interactive Streamlit app that cleans and explores NBA game data, then trains several classifiers to predict whether the home team wins a game. Final project of my Python Programming course.

## Data

[NBA games data](https://www.kaggle.com/datasets/nathanlauga/nba-games) (Kaggle), included here as `NBA dataset.zip`:

| File | Content |
|---|---|
| `games.csv` | One row per game since 2004: teams, points, shooting percentages, assists, rebounds, home win yes/no |
| `games_details.csv` | Player statistics for each game |
| `ranking.csv` | Daily team standings, East and West |
| `teams.csv` | The 30 NBA teams |

## What the app shows

1. **Data cleaning:** dropping unused columns, joining player and team tables, removing duplicates and teams that no longer exist, filling missing values.
2. **Charts:** points by starting position, three-pointers vs total points, star player comparisons, home vs away results, shooting percentages at home and away.
3. **Prediction:** predicts `HOME_TEAM_WINS` from ten game statistics (field goal %, free throw %, three-point %, assists and rebounds for the home and away teams). Models compared: random forest, XGBoost, CatBoost, Gaussian Naive Bayes, logistic regression, KNN and a tuned decision tree (grid search on balanced accuracy).

## Files

| File | Description |
|---|---|
| `Streamlit Project Programming.py` | The Streamlit app |
| `NBA dataset.zip` | The four CSV files |
| `Final Project Programming.zip` | The Jupyter notebook version of the analysis |

## How to run

The app was written with pandas 1.x and needs Python 3.8–3.11.

```bash
git clone https://github.com/Amirhatamian/Python-Programming-Project.git
cd Python-Programming-Project
unzip "NBA dataset.zip"
pip install "pandas<2" numpy matplotlib seaborn plotly scikit-learn xgboost catboost streamlit
streamlit run "Streamlit Project Programming.py"
```

## Tools

Python, pandas, Streamlit, Plotly, Seaborn, Matplotlib, scikit-learn, XGBoost, CatBoost
