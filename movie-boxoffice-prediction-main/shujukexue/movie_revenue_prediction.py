import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import pickle
from datetime import datetime, timedelta

np.random.seed(42)

genres = ['Action', 'Adventure', 'Animation', 'Comedy', 'Crime', 'Documentary', 'Drama', 'Fantasy', 'Horror', 'Romance', 'Science Fiction', 'Thriller']
languages = ['en', 'zh', 'ja', 'ko', 'fr', 'es', 'de']
production_companies = ['Disney', 'Warner Bros', 'Universal', 'Paramount', 'Sony', 'Fox', 'Netflix', 'Amazon', 'MGM', 'Lionsgate']

num_movies = 500

movie_data = {
    'Title': [f'Movie_{i+1}' for i in range(num_movies)],
    'Budget': np.random.randint(10000000, 300000000, num_movies),
    'Popularity': np.random.uniform(1, 100, num_movies),
    'Vote_Average': np.random.uniform(4.0, 9.5, num_movies),
    'Vote_Count': np.random.randint(100, 100000, num_movies),
    'Original_Language': np.random.choice(languages, num_movies),
    'Runtime': np.random.randint(80, 200, num_movies),
    'Genres': np.random.choice(genres, num_movies),
    'Production_Companies': np.random.choice(production_companies, num_movies),
    'Release_Date': [datetime(2010, 1, 1) + timedelta(days=np.random.randint(0, 365*14)) for _ in range(num_movies)]
}

df = pd.DataFrame(movie_data)

df['Revenue'] = (
    df['Budget'] * (0.5 + df['Popularity'] / 100 * 0.8) * 
    (1 + (df['Vote_Average'] - 5) * 0.15) *
    (1 + np.log1p(df['Vote_Count']) / 10) *
    (1 + (df['Runtime'] - 100) / 200)
) + np.random.normal(0, df['Budget'] * 0.1, num_movies)

df['Revenue'] = df['Revenue'].apply(lambda x: max(x, 1000000))

print("数据集前5行预览：")
print(df.head())
print("\n数据集基本信息：")
print(df.info())
print("\n数据集统计摘要：")
print(df.describe())

plt.figure(figsize=(12, 8))
sns.histplot(df['Revenue'] / 1e6, bins=30, kde=True)
plt.title('票房收入分布（百万美元）')
plt.xlabel('票房收入（百万美元）')
plt.ylabel('电影数量')
plt.savefig('revenue_distribution.png')
plt.close()

plt.figure(figsize=(12, 8))
sns.scatterplot(x='Budget', y='Revenue', data=df)
plt.title('预算与票房关系')
plt.xlabel('预算（美元）')
plt.ylabel('票房（美元）')
plt.savefig('budget_revenue.png')
plt.close()

df['Release_Year'] = df['Release_Date'].dt.year
df['Release_Month'] = df['Release_Date'].dt.month

X = df.drop(['Title', 'Revenue', 'Release_Date'], axis=1)
y = df['Revenue']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

numeric_features = ['Budget', 'Popularity', 'Vote_Average', 'Vote_Count', 'Runtime', 'Release_Year', 'Release_Month']
categorical_features = ['Original_Language', 'Genres', 'Production_Companies']

preprocessor = ColumnTransformer(
    transformers=[
        ('num', StandardScaler(), numeric_features),
        ('cat', OneHotEncoder(drop='first'), categorical_features)
    ])

models = {
    'Linear Regression': LinearRegression(),
    'Random Forest': RandomForestRegressor(n_estimators=100, random_state=42),
    'Gradient Boosting': GradientBoostingRegressor(n_estimators=100, random_state=42)
}

results = {}
best_model = None
best_r2 = -float('inf')

for model_name, model in models.items():
    pipeline = Pipeline([
        ('preprocessor', preprocessor),
        ('regressor', model)
    ])
    
    pipeline.fit(X_train, y_train)
    y_pred = pipeline.predict(X_test)
    
    mae = mean_absolute_error(y_test, y_pred)
    mse = mean_squared_error(y_test, y_pred)
    rmse = np.sqrt(mse)
    r2 = r2_score(y_test, y_pred)
    
    results[model_name] = {
        'MAE': mae,
        'MSE': mse,
        'RMSE': rmse,
        'R2': r2,
        'model': pipeline
    }
    
    if r2 > best_r2:
        best_r2 = r2
        best_model = pipeline
    
    print(f"\n{model_name} 模型评估结果：")
    print(f"  MAE: {mae:,.2f}")
    print(f"  MSE: {mse:,.2f}")
    print(f"  RMSE: {rmse:,.2f}")
    print(f"  R2: {r2:.4f}")

print("\n" + "="*50)
print("模型性能对比：")
print("="*50)
comparison_df = pd.DataFrame(results).T[['MAE', 'RMSE', 'R2']]
print(comparison_df)

plt.figure(figsize=(12, 6))
comparison_df['R2'].plot(kind='bar', color=['blue', 'green', 'orange'])
plt.title('各模型R2分数对比')
plt.ylabel('R2分数')
plt.ylim(0, 1)
plt.savefig('model_comparison.png')
plt.close()

plt.figure(figsize=(10, 10))
sns.heatmap(df[numeric_features + ['Revenue']].corr(), annot=True, cmap='coolwarm')
plt.title('特征相关性矩阵')
plt.savefig('correlation_matrix.png')
plt.close()

with open('movie_revenue_model.pkl', 'wb') as f:
    pickle.dump(best_model, f)

print("\n模型已保存为 movie_revenue_model.pkl")

sample_movie = pd.DataFrame({
    'Budget': [150000000],
    'Popularity': [85.5],
    'Vote_Average': [8.2],
    'Vote_Count': [50000],
    'Original_Language': ['en'],
    'Runtime': [135],
    'Genres': ['Action'],
    'Production_Companies': ['Disney'],
    'Release_Year': [2024],
    'Release_Month': [7]
})

predicted_revenue = best_model.predict(sample_movie)
print(f"\n示例电影预测票房：${predicted_revenue[0]:,.2f}")