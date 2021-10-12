```python
# Primary libraries
import pandas as pd
import numpy as np
import statsmodels.api as sm
import statsmodels.formula.api as smf

# Custom functions
from src import helpers
```

# Linear Regression Practice

A dataset called `diabetes.csv` is stored in the `data` folder for this repository. 

In the cell below, read in the dataset using pandas, and output the head of the dataframe. Assign the dataframe to the variable `df`.


```python
# Your code here
```

**Each row in this dataset represents a patient with diabetes.** 

For this assignment, the variables of focus will be:
* age
* sex
* bmi
* bp
* target

<details>
    <summary>
        <i>Click here to view the documentation for the dataset
        </i>
    </summary>
    <h1>Diabetes dataset</h1>
    <p>


Ten baseline variables, age, sex, body mass index, average blood
pressure, and six blood serum measurements were obtained for each of n =
442 diabetes patients, as well as the response of interest, a
quantitative measure of disease progression one year after baseline.

**Data Set Characteristics:**

  :Number of Instances: 442

  :Number of Attributes: First 10 columns are numeric predictive values

  :Target: Column 11 is a quantitative measure of disease progression one year after baseline

  :Attribute Information:
      - age     age in years
      - sex
      - bmi     body mass index
      - bp      average blood pressure
      - s1      tc, total serum cholesterol
      - s2      ldl, low-density lipoproteins
      - s3      hdl, high-density lipoproteins
      - s4      tch, total cholesterol / HDL
      - s5      ltg, possibly log of serum triglycerides level
      - s6      glu, blood sugar level

Note: Each of these 10 feature variables have been mean centered and scaled by the standard deviation times `n_samples` (i.e. the sum of squares of each column totals 1).

Source URL:
https://www4.stat.ncsu.edu/~boos/var.select/diabetes.html

For more information see:
Bradley Efron, Trevor Hastie, Iain Johnstone and Robert Tibshirani (2004) "Least Angle Regression," Annals of Statistics (with discussion), 407-499.
(https://web.stanford.edu/~hastie/Papers/LARS/LeastAngle_2002.pdf)
    
  </p>
</details>


```python
df = df[['target', 'age', 'sex', 'bmi', 'bp']]
df.head(2)
```


```python
# Run this cell unchanged
helpers.independent.display()
```

**For our first model, let's figure out which column is most correlated with the target.**

In the cell below,

* Identify the most correlated feature with the target column.
* Assign the name of the column to the variable `most_correlated`.


```python
# Your code here
```


```python
# Run this cell unchanged!
helpers.correlation.display()
```


```python
# Run this cell unchanged!
helpers.correlation_strong.display()
```

Let's create a model using the most correlated feature as the only predictor.

## There are two main ways of creating a linear regression model when using `statsmodels`. 
### 1. The pythonic way
   - For this approach you will typically see statsmodels imported with the line `import statsmodels.api as sm`
   - Using this approach, you create the model by passing the actual data objects into the model like so..
        
        --------------
        ```python
        model = sm.OLS(df.target, df.bmi)
        model_results = model.fit()
        ```
        --------------
       
       
   - This approach can be handy when you have a lot a features and do not wish to type the name of each column because you can pass in an entire dataframe of predictors.
   - One small annoyance with this approach is that the model does not use an intercept by default so you typically have to add the intercept manually
    
    
    --------------
    ```python
    model = sm.OLS(df.target, df[['bmi']].assign(intercept=1))
    model_results = model.fit()
    ```
    
    ----------------
    
### 2. The `R` formula way
   - For this approach you will typically see statsmodels imported with the line `import statsmodels.formula.api as smf`
   - Using this approach, you write your linear equation as a string with the following format:
    
```python
'{dependent_variable} ~ {independent_variable_1} + ... {indepdendent_variable_n}'
```
        
   - In this case, with a dependent variable of `target` and a single independent variable of `bmi`, our formula looks like this:
   ```python
    'target ~ bmi'
    ```
   - And the full modeling code looks like this:
   
   --------
   
```python
formula = 'target ~ bmi'
model = smf.ols(formula, data=df)
model_results = model.fit()
```
    
   --------
    
   - Using this approach, the intercept is added by default
   - One downside of this approach is that writing the formula can be a little cumbersome when you have a lot of features
   
   
## tl;dr There are multiple ways of creating a model, but either option works perfectly fine. 

In this notebook, we will focus on using the `R` formula method for the following reasons:
1. This bootcamp focuses primarily on  *Ordinary Least Squares Linear Regression*, but there are some more advanced versions of linear regression in statsmodels that are only supported by the formula approach. Because of this, familiarity with the formula technique is highly encouraged.
2. The formula approach adds an intercept term by default which is extremely convenient!

**In the cell below, write the formula for our first linear regression model and assign the string to a variable called `formula1`.**


```python
# Your code here
```

Let's fit the model and interpret the results!


```python
# Your code here
```

**Using the table above, we have all the information we need to write a linear equation.**

Intercept = 152.1335

Slope = 949.4353

Linear Equation:  $target = 152.1335 + 949.4353bmi$

## Interpret the numbers

#### Interpreting the intercept


```python
# Run this cell unchanged
helpers.intercept.display()
```

#### Interpret the slope


```python
# Run this cell unchanged
helpers.slope.display()
```

## Evaluate the model

In the cell below,
1. Import `r2_score` and `mean_squared_error` from the sklearn metrics module.
2. Calculate the r2 and RMSE for the model.


```python
# Import r2_score and mean_squared_error
# Your code here

# Calculate r2
# Replace None with your code
model1_r2 = None

# Calculate RMSE
# Replace None with your code
model1_rmse = None

# Print Results
print('Model R^2: ', model1_r2)
print('Model RMSE:', model1_rmse)
```

### Interpret the r2 and RMSE metrics

YOUR ANSWER HERE

### Check assumptions of Linear Regression

**Investigating Linearity**

First, let's check whether the linearity assumption holds.


```python
# Import matplotlib using the standard alias
# Your code here

# Generate model predictions
# Replace None with your code
preds = None

# Create a matplotlib subplot
# Replace None with your code
fig, ax = None, None

# Using numpy.arange create a line that
# begins with the minimum target value
# and ends with the maximum target value
perfect_line = None

# Plot the `perfect_line`
# Set the label to the string "Perfect Fit"
# Your code here

# Plot a scatter plot setting the xaxis to 
# the true target values and the yaxis to
# the predicted valued
# Your code here

# Set the xlabel to the string "Actual Blood Pressure"
# Your code here

# Set the ylabel to the string "Predicted Blood Pressure"
# Your code here

# Activate the legend for the matplotlib axis
# Your code here
```

Are you violating the linearity assumption?

YOUR ANSWER HERE

**Investigating Normality**


```python
from scipy.stats import norm

fig, axes = plt.subplots(1,2, figsize=(20,5))
residuals = preds-df.target
sm.graphics.qqplot(residuals, dist=norm, line='45', fit=True, ax=axes[0])
axes[1].hist(residuals);
```

**Investigating Equal Variance**


```python
fig, ax = plt.subplots(figsize=(15,6))

ax.scatter(preds, residuals, alpha=0.5)
ax.plot(preds, [0 for i in range(len(df))])
ax.set_xlabel("Predicted Value")
ax.set_ylabel("Predicted Value - Actual");
```

Are you violating the equal variance assumption?

YOUR ANSWER HERE

### Multiple Linear Regression

In the cell below, define fit a linear regression using `bmi` and `bp` as independent variables.


```python
formula2 = None

model2 = None
model2.summary()
```

### Define a function to calculate the `r2` and `mean squared error`


```python
def metrics(model, target):
    # Generate model predictions
    None
    # Calcuate r2
    None
    # Calculate rmse
    None
    # Return r2 and rmse
    None
```


```python
model2_r2, model2_rmse = None, None

# Print Results
print('Model1 R^2:', model1_r2)
print('Model2 RMSE:', model2_rmse)
print('-------------------------------')

print('Model R^2: ', model2_r2)
print('Model RMSE:', model2_rmse)
```

### Define a function to plot linearity


```python
def linearity(model, target):
    # Generate model predictions
    None
    # Create a matplotlib subplot object
    None
    # Generate a line beginning with target min and
    # ending with target max
    None
    # Plot the perfect line
    None
    # Plot a scatter plot with the target as the xaxis and preds as the yaxis
    None
    # Set the x and y axis labels
    None
    # Activate the axis legend
    None
```


```python
linearity(model2, df.target)
```

### Define a function to plot residual normality.


```python
def normality(model, target):
    # Initialize a matplotlib subplot
    None
    # Generate model predictions
    None
    # Calculate error
    None
    # Plot the residuals as a qqplot
    None
    # Plot a histogram
    None
```


```python
normality(model2, df.target)
```

### Define a function to plot equal_variance


```python
def equal_var(model, target):
    # Generate model predictions
    None
    # Calculate error
    None
    # Initialize a matplotlib subplot
    None
    # Plot scatter plot of predictions vs error
    None
    # Plot a horizontal line set at 0
    None
    # Set the x and y axis labels
    None
    None
```


```python
equal_var(model2, df.target)
```

### Multicolinearity 


```python
from statsmodels.stats.outliers_influence import variance_inflation_factor

rows = df[["bmi", "bp"]].values

vif_df = pd.DataFrame()
vif_df["VIF"] = [variance_inflation_factor(rows, i) for i in range(2)]
vif_df["feature"] = ["bmi", "bp"]

vif_df
```

### Multi-colinearity Function


```python
def vif(features_df):
    rows = features_df.values
    vif_df = pd.DataFrame()
    vif_df["VIF"] = [variance_inflation_factor(rows, i) for i in range(features_df.shape[1])]
    vif_df["feature"] = features_df.columns
    return vif_df
```


```python
vif(df[['bmi', 'bp']])
```


```python

```
