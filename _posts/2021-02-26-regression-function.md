---
layout: post
title:  "Regression function"
date:   2021-02-18 16:39:48 +0000
categories: jekyll update
---

Here, I wrote a function to run bivariate and multivariate linear regression using OLS, visualise the regressions in both 2D and 3D and add the option to plot a coefficient plot.

This is a post in progress. More will be added later :)

{% highlight python %}
# Define a function to perform OLS bivariate and multivariate linear regression
def all_regression(df, regressors, predictor, coef_plots=False): # regressors is a list argument, or if bivariate, just a single string
    """
    Modules to import:
    import pandas as pd
    import numpy as np
    import matplotlib.pyplot as plt
    import statsmodels.api as sm
    import seaborn as sns
    from mpl_toolkits.mplot3d import Axes3D
    """
    # Setting up the design matrix
    X = df[regressors]
    X = sm.add_constant(X)

    # Setting up predictor vector
    y = df[predictor]

    # Create a model and fit
    model = sm.OLS(y, X, missing='drop')
    results = model.fit()
    
    # Visualising observations and model (different visualisations depending if bi or multi)
    if len(regressors) == 1: # if bivariate regressors argument inputted as single item list
        raise TypeError("If using all_regression() for bivariate regression, 'regressors' argument must be a string")
    
    # if bivariate with valid argument type
    elif type(regressors) == str:
        
        # Setting up the 2D space 
        fig, ax = plt.subplots()

        # Plotting the observations
        ax.plot(X[regressors], y, 'o', c = 'blue', label = 'Data')

        # Plotting the OLS regression line
        ax.plot(X[regressors], results.fittedvalues, '--', c = 'red', label = 'OLS')
        ax.set_title(f"{regressors} against {predictor}")
        ax.set_xlabel(regressors)
        ax.set_ylabel(predictor)

        # Adding a legend and showing the despined plot
        ax.legend(loc = 'upper left')
        sns.despine()
        plot = plt.show()
        
        # Plotting coefficient plot if True
        if coef_plots == True:
            # Create a pandas series of errors
            err_series = results.params - results.conf_int()[0]

            # Create dataframe of coefficients, errors and variable names
            coef_df = pd.DataFrame({'coef': results.params.values[1:],
                        'err': err_series.values[1:],
                        'varname': err_series.index.values[1:]
                       })
            fig1, ax1 = plt.subplots(figsize=(8, 5))

            # Plot box and whisker on plot
            coef_df.plot(x='varname', y='coef', kind='bar', 
                         ax=ax1, color='none', 
                         yerr='err', legend=False)
            ax1.set_ylabel('')
            ax1.set_xlabel('')
            ax1.scatter(x=np.arange(coef_df.shape[0]), 
                       marker='s', s=120, 
                       y=coef_df['coef'], color='black')
            ax1.axhline(y=0, linestyle='--', color='black', linewidth=4)
            ax1.xaxis.set_ticks_position('none')
            _ = ax1.set_xticklabels([regressors], 
                                   rotation=0, fontsize=16)
            plot1 = plt.show()

            return results.summary(), plot, plot1
        else:
            return results.summary(), plot
    
    # else (if multivariate with a maximum of 2 regressors)
    elif len(regressors) <= 2: 
        
        # Setting up 3D space
        fig = plt.figure()
        ax = fig.add_subplot(111, projection='3d')

        # Plotting observations and naming axis in 3D space
        x1 = df[regressors[0]]
        x2 = df[regressors[1]]
        ax.set_xlabel(regressors[0])
        ax.set_ylabel(regressors[1])
        ax.set_zlabel(predictor)
        ax.scatter(x1, x2, y, c=np.linalg.norm([x1, x2, y], axis=0))

        # Creating arrays to form meshgrid
        x_surf = np.linspace(min(x1), max(x1), 100)
        y_surf = np.linspace(min(x2), max(x2), 100)
        x_surf, y_surf = np.meshgrid(x_surf, y_surf)
        
        # Ravelling arrays into dataframe and using them to plot fitted values
        onlyX = pd.DataFrame({regressors[0]: x_surf.ravel(), regressors[1]: y_surf.ravel()})
        onlyX = np.array(sm.add_constant(onlyX))
        fittedY = results.predict(exog=onlyX)
        ax.plot_surface(x_surf, y_surf, fittedY.reshape(x_surf.shape), color='Red', alpha=0.5)
        plot = plt.show()

        if coef_plots == True:
            # Create a pandas series of errors
            err_series = results.params - results.conf_int()[0]

            # Create dataframe of coefficients, errors and variable names
            coef_df = pd.DataFrame({'coef': results.params.values[1:],
                        'err': err_series.values[1:],
                        'varname': err_series.index.values[1:]
                       })
            fig1, ax1 = plt.subplots(figsize=(8, 5))

            # Plot box and whisker on plot
            coef_df.plot(x='varname', y='coef', kind='bar', 
                         ax=ax1, color='none', 
                         yerr='err', legend=False)
            ax1.set_ylabel('')
            ax1.set_xlabel('')
            ax1.scatter(x=np.arange(coef_df.shape[0]), 
                       marker='s', s=120, 
                       y=coef_df['coef'], color='black')
            ax1.axhline(y=0, linestyle='--', color='black', linewidth=4)
            ax1.xaxis.set_ticks_position('none')
            _ = ax1.set_xticklabels(regressors, 
                                   rotation=0, fontsize=16)
            plot1 = plt.show()

            return results.summary(), plot, plot1
        else:
            return results.summary(), plot
    
    # else (if too many regressors to visualise)
    else:         
        if coef_plots == True:
            # Create a pandas series of errors
            err_series = results.params - results.conf_int()[0]
            
            # Create dataframe of coefficients, errors and variable names
            coef_df = pd.DataFrame({'coef': results.params.values[1:],
                        'err': err_series.values[1:],
                        'varname': err_series.index.values[1:]
                       })
            fig1, ax1 = plt.subplots(figsize=(8, 5))
            
            # Plot box and whisker on plot
            coef_df.plot(x='varname', y='coef', kind='bar', 
                         ax=ax1, color='none', 
                         yerr='err', legend=False)
            ax1.set_ylabel('')
            ax1.set_xlabel('')
            ax1.scatter(x=np.arange(coef_df.shape[0]), 
                       marker='s', s=120, 
                       y=coef_df['coef'], color='black')
            ax1.axhline(y=0, linestyle='--', color='black', linewidth=4)
            ax1.xaxis.set_ticks_position('none')
            _ = ax1.set_xticklabels(regressors, 
                                   rotation=0, fontsize=16)
            plot1 = plt.show()
            
            return results.summary(), plot1, f"Sorry, I can't visualise {len(regressors)} regressors!"
        else:
            return results.summary(), f"Sorry, I can't visualise {len(regressors)} regressors!"
{% endhighlight %}