import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import duckdb
import scipy
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score
# go
# Read and load data 
df = pd.read_csv("C:/Users/lenovo/Downloads/nwsl-team-stats.csv", sep = ',')
df = pd.DataFrame(df)
print(df)

"""
Motivation
The National Women’s Soccer League (NWSL) is the top professional women’s soccer league in the United States. 
While a team’s record ultimately determines their ranking, goal differential (goals scored - goals conceded) 
is often a better indicator of a team’s ability. But what aspects of a team’s performance are related to their
goal differential? The NWSL records a variety of statistics describing a team’s performance, 
such as the percentage of time they maintain possession, percentage of shots on target, etc. 
With this dataset, you can explore variation between teams as well as which statistics are
relevant predictor variables of goal differential.
"""

df.columns # Features in the data
"""
 Variable	Description
team_name -	Name of NWSL team
season	  -  Regular season year of team’s statistics
games_played -	Number of games team played in season
goal_differential -	Goals scored - goals conceded
goals -	Number of goals scores
goals_conceded -	Number of goals conceded
cross_accuracy	- Percent of crosses that were successful
goal_conversion_pct -	Percent of shots scored
pass_pct	- Pass accuracy
pass_pct_opposition_half	- Pass accuracy in opposition half
possession_pct	- Percentage of overall ball possession the team had during the season
shot_accuracy -	Percentage of shots on target
tackle_success_pct -	Percent of successful tackles  
    """



columns_to_drop = ['team_name','season','goal_differential']
X = df.drop(columns=columns_to_drop)
y = df["goal_differential"]

model = LinearRegression()
model.fit(X, y)

# Make predictions
y_pred = model.predict(X)

# Evaluate the model
mse = np.mean((y - y_pred) ** 2)
r2 = model.score(X, y)

print(f'Mean Squared Error: {mse}')
print(f'R-squared: {r2}')

# Visualize the results
plt.scatter(y, y_pred)
plt.xlabel('Actual Goal Diffenence')
plt.ylabel('Predicted Goal Difference')
plt.title('Actual vs Predicted Goal Difference')
plt.show()

# Optionally, print model coefficients
coefficients = pd.DataFrame(model.coef_, X.columns, columns=['Coefficient'])
print(coefficients)