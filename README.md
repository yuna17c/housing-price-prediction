# housing-price-prediction
In this notebook, I explored data from housing price prediction (provided [here](https://www.kaggle.com/datasets/harishkumardatalab/housing-price-prediction)) to create a model that accurately estimates house prices. 
I used multiple tools such as Pandas, Matplotlib, NumPy, and Seaborn to visualize and summarize the data. I used stacked regressions to create a prediction model. 

The notebook has three sections: 
1) Data Observations & Visualizations
In this section, I looked at the distributions of each feature and tried to figure out important features. I also visualized the relationships and characteristics of features using a heatmap, density plots, scatter plots, and boxplots.
2) Data Manipulation
I dealt with missing data and outliers. I also transformed some skewed features. I also added and removed features as needed. 
3) Modelling
I created a stacked regression model to estimate house prices. I used regressors such as XGBoost, LightGBM, Ridge, and Random Forest. The final RMSLE score on train data is 0.007755.

The intention of this project is to learn and I took inspiration from this [Kaggle notebook](https://www.kaggle.com/datasets/harishkumardatalab/housing-price-prediction).
