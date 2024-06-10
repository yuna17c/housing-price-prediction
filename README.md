# housing-price-prediction
The streamlit app is deployed at this [link](https://yuna17c-housing-price-prediction-dashboard-hecnzz.streamlit.app/).

Or you can open the app locally by running
`streamlit run dashboard.py`

The notebook has three sections.
1) Data Observations & Visualizations:
In this section, I looked at the distributions of each feature and tried to figure out important features. I also visualized the relationships and characteristics of features using a heatmap, density plots, scatter plots, and boxplots.
2) Data Manipulation:
I dealt with missing data and outliers. I also transformed some skewed features and added/removed features as needed. 
3) Modelling:
I used a stacked regression model and ensemble method to estimate house prices. Elastic net, kernel ridge, and gradient boosting are used as base models, and lasso is used as the meta model in stacked regression. The stacked model and XGBoost model are then ensembled to make a prediction. The final RMSE on the stacked model is 0.0803 and the RMSE on the XGBoost model is 0.0880,
