# housing-price-prediction
Open the app by running the following command
`streamlit run dashboard.py`

The notebook has three sections: 
1) Data Observations & Visualizations
In this section, I looked at the distributions of each feature and tried to figure out important features. I also visualized the relationships and characteristics of features using a heatmap, density plots, scatter plots, and boxplots.
2) Data Manipulation
I dealt with missing data and outliers. I also transformed some skewed features and added/removed features as needed. 
3) Modelling
I created a stacked regression model to estimate house prices. I used regressors such as XGBoost, LightGBM, Ridge, and Random Forest. The final RMSLE score on train data is 0.007755.
