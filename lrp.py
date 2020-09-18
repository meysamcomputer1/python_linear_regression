import pandas
from pandas import DataFrame
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression 

data = pandas.read_csv('cost_revenue_clean.csv')
print(data.describe())

X = DataFrame(data, columns=['production_budget_usd'])
Y = DataFrame(data, columns=['worldwide_gross_usd'])

regression = LinearRegression()
regression.fit(X, Y)
# show slope
print("Slope is: " + str(regression.coef_))
# show constant
print("Constant is : " + str(regression.intercept_))
    
plt.scatter(X, Y, alpha=0.3)
plt.title('Film Cost vs Global Revenue')
plt.xlabel('Production budget $')
plt.ylabel('Worldwide Gross $')
plt.xlim(0, 350000000)
plt.ylim(0, 3000000000)

plt.plot(X, regression.predict(X), color='red', linewidth=2)

plt.show()