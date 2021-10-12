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
df = pd.read_csv('data/diabetes.csv')
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




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>target</th>
      <th>age</th>
      <th>sex</th>
      <th>bmi</th>
      <th>bp</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>151.0</td>
      <td>0.038076</td>
      <td>1</td>
      <td>0.061696</td>
      <td>0.021872</td>
    </tr>
    <tr>
      <th>1</th>
      <td>75.0</td>
      <td>-0.001882</td>
      <td>0</td>
      <td>-0.051474</td>
      <td>-0.026328</td>
    </tr>
  </tbody>
</table>
</div>




```python
# Run this cell unchanged
helpers.independent.display()
```


    VBox(children=(Output(outputs=({'output_type': 'display_data', 'data': {'text/plain': '<IPython.core.display.M…


**For our first model, let's figure out which column is most correlated with the target.**

In the cell below,

* Identify the most correlated feature with the target column.
* Assign the name of the column to the variable `most_correlated`.


```python
most_correlated = df.corr().sort_values('target').iloc[-2].name
```


```python
# Run this cell unchanged!
helpers.correlation.display()
```


    VBox(children=(Output(outputs=({'output_type': 'display_data', 'data': {'text/plain': '<IPython.core.display.M…



```python
# Run this cell unchanged!
helpers.correlation_strong.display()
```


    VBox(children=(Output(outputs=({'output_type': 'display_data', 'data': {'text/plain': '<IPython.core.display.M…


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
formula1 = 'target ~ bmi'
```

Let's fit the model and interpret the results!


```python
model1 = smf.ols(formula1, df).fit()
model1.summary().tables[1]
```




<table class="simpletable">
<tr>
      <td></td>         <th>coef</th>     <th>std err</th>      <th>t</th>      <th>P>|t|</th>  <th>[0.025</th>    <th>0.975]</th>  
</tr>
<tr>
  <th>Intercept</th> <td>  152.1335</td> <td>    2.974</td> <td>   51.162</td> <td> 0.000</td> <td>  146.289</td> <td>  157.978</td>
</tr>
<tr>
  <th>bmi</th>       <td>  949.4353</td> <td>   62.515</td> <td>   15.187</td> <td> 0.000</td> <td>  826.570</td> <td> 1072.301</td>
</tr>
</table>



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


    VBox(children=(Output(outputs=({'output_type': 'display_data', 'data': {'text/plain': '<IPython.core.display.M…


#### Interpret the slope


```python
# Run this cell unchanged
helpers.slope.display()
```


    VBox(children=(Output(outputs=({'output_type': 'display_data', 'data': {'text/plain': '<IPython.core.display.M…


## Evaluate the model

In the cell below,
1. Import `r2_score` and `mean_squared_error` from the sklearn metrics module.
2. Calculate the r2 and RMSE for the model.


```python
# Import r2_score and mean_squared_error
from sklearn.metrics import r2_score, mean_squared_error

# Calculate r2
model1_r2 = r2_score(df.target, model1.predict())

# Calculate RMSE
model1_rmse = mean_squared_error(df.target, model1.predict(), squared=False)

# Print Results
print('Model R^2: ', model1_r2)
print('Model RMSE:', model1_rmse)
```

    Model R^2:  0.3439237602253803
    Model RMSE: 62.37352471570989


### Interpret the r2 and RMSE metrics



The regression model `target ~ bmi` explains about 34% of the variance in the dependent variable. The predictions generated by this model have an average error 62.

### Check assumptions of Linear Regression

**Investigating Linearity**

First, let's check whether the linearity assumption holds.

Are you violating the linearity assumption?


Yes. The predictions generated by this model are deviating from a linear relationship. At low blood pressure levels, the model appears to be mostly overpredicting, and is dramatically underpredicting beyond a blood pressure of ~125.

**Investigating Normality**


```python
from scipy.stats import norm

fig, axes = plt.subplots(1,2, figsize=(20,5))
residuals = preds-df.target
sm.graphics.qqplot(residuals, dist=norm, line='45', fit=True, ax=axes[0])
axes[1].hist(residuals);
```

**Investigating Equal Variance**

Are you violating the equal variance assumption?


This doesn't look to bad, but there is some evidence of heteroscedasticty. As we saw from this linearity plot, our model is mostly overestimating or underestimating, which is reflected in this plot. The data appears to be a bit more densly distributed at lower values and it becoming more widley distributes as the predicted values increase. The overall variance, however seems to be largly consistent. As the predicted values increase, we do not see a widening of distribution but moreso a change in the mean of the residuals (This again, is better demonstrated by our violation of linearity)

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
    preds = model.predict()
    # Calcuate r2
    r2 = r2_score(target, preds)
    # Calculate rmse
    rmse = mean_squared_error(target, preds, squared=False)
    # Return r2 and rmse
    return r2, rmse
```


```python
model2_r2, model2_rmse = metrics(model2, df.target)

# Print Results
print('Model1 R^2:', model1_r2)
print('Model2 RMSE:', model2_rmse)
print('-------------------------------')

print('Model R^2: ', model2_r2)
print('Model RMSE:', model2_rmse)
```

    Model1 R^2: 0.3439237602253803
    Model2 RMSE: 59.84718212461125
    -------------------------------
    Model R^2:  0.39599414313715675
    Model RMSE: 59.84718212461125


### Define a function to plot linearity


```python
def linearity(model, target):
    # Generate model predictions
    preds = model.predict()
    # Create a matplotlib subplot object
    fig, ax = plt.subplots()
    # Generate a line beginning with target min and
    # ending with target max
    perfect_line = np.arange(target.min(), target.max())
    # Plot the perfect line
    ax.plot(perfect_line, linestyle="--", color="orange", label="Perfect Fit")
    # Plot a scatter plot with the target as the xaxis and preds as the yaxis
    ax.scatter(target, preds, alpha=0.5)
    # Set the x and y axis labels
    ax.set_xlabel("Actual Blood Pressure")
    ax.set_ylabel("Predicted Blood Pressure")
    # Activate the axis legend
    ax.legend();
```


```python
linearity(model2, df.target)
```

### Define a function to plot residual normality.


```python
normality(model2, df.target)
```

### Define a function to plot equal_variance


```python
def equal_var(model, target):
    # Generate model predictions
    preds = model.predict()
    # Calculate error
    residuals = preds - target
    # Initialize a matplotlib subplot
    fig, ax = plt.subplots(figsize=(15,6))
    # Plot scatter plot of predictions vs error
    ax.scatter(preds, residuals, alpha=0.5)
    # Plot a horizontal line set at 0
    ax.plot(preds, [0 for i in range(len(df))])
    # Set the x and y axis labels
    ax.set_xlabel("Predicted Value")
    ax.set_ylabel("Predicted Value - Actual");
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
