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
import streamlit as st

# --------------------------
# 页面配置
# --------------------------
st.set_page_config(
    page_title="电影票房预测系统",
    page_icon="🎬",
    layout="wide"
)

st.title("🎬 电影票房收入预测应用")
st.markdown("基于机器学习的电影票房预测模型，支持数据可视化、模型训练与实时预测")

# --------------------------
# 固定随机种子
# --------------------------
np.random.seed(42)

# --------------------------
# 侧边栏：功能选择
# --------------------------
st.sidebar.header("功能菜单")
menu = st.sidebar.radio(
    "选择功能",
    ["数据集概览", "数据可视化", "模型训练与评估", "票房预测"]
)

# --------------------------
# 生成数据集（与原代码一致）
# --------------------------
@st.cache_data  # 缓存数据，避免重复生成
def generate_data():
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
    df['Release_Year'] = df['Release_Date'].dt.year
    df['Release_Month'] = df['Release_Date'].dt.month
    
    return df

df = generate_data()

# --------------------------
# 1. 数据集概览
# --------------------------
if menu == "数据集概览":
    st.subheader("📊 数据集概览")
    st.dataframe(df.head(10), use_container_width=True)
    
    st.subheader("数据集基本信息")
    buffer = pd.DataFrame({
        "特征": df.columns,
        "数据类型": df.dtypes,
        "非空值数量": df.count().values
    })
    st.dataframe(buffer, use_container_width=True)
    
    st.subheader("数据统计摘要")
    st.dataframe(df.describe(), use_container_width=True)

# --------------------------
# 2. 数据可视化
# --------------------------
elif menu == "数据可视化":
    st.subheader("📈 数据可视化分析")
    
    tab1, tab2, tab3 = st.tabs(["票房分布", "预算与票房关系", "特征相关性"])
    
    with tab1:
        fig, ax = plt.subplots(figsize=(10, 6))
        sns.histplot(df['Revenue'] / 1e6, bins=30, kde=True, ax=ax)
        ax.set_title('票房收入分布（百万美元）')
        ax.set_xlabel('票房收入（百万美元）')
        ax.set_ylabel('电影数量')
        st.pyplot(fig)
        
    with tab2:
        fig, ax = plt.subplots(figsize=(10, 6))
        sns.scatterplot(x='Budget', y='Revenue', data=df, ax=ax)
        ax.set_title('预算与票房关系')
        ax.set_xlabel('预算（美元）')
        ax.set_ylabel('票房（美元）')
        st.pyplot(fig)
        
    with tab3:
        numeric_features = ['Budget', 'Popularity', 'Vote_Average', 'Vote_Count', 'Runtime', 'Release_Year', 'Release_Month']
        fig, ax = plt.subplots(figsize=(12, 10))
        sns.heatmap(df[numeric_features + ['Revenue']].corr(), annot=True, cmap='coolwarm', ax=ax)
        ax.set_title('特征相关性矩阵')
        st.pyplot(fig)

# --------------------------
# 3. 模型训练与评估
# --------------------------
elif menu == "模型训练与评估":
    st.subheader("🤖 模型训练与性能评估")
    
    # 数据准备
    X = df.drop(['Title', 'Revenue', 'Release_Date'], axis=1)
    y = df['Revenue']
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    numeric_features = ['Budget', 'Popularity', 'Vote_Average', 'Vote_Count', 'Runtime', 'Release_Year', 'Release_Month']
    categorical_features = ['Original_Language', 'Genres', 'Production_Companies']
    
    # 预处理管道
    preprocessor = ColumnTransformer(
        transformers=[
            ('num', StandardScaler(), numeric_features),
            ('cat', OneHotEncoder(drop='first'), categorical_features)
        ])
    
    # 模型定义
    models = {
        'Linear Regression': LinearRegression(),
        'Random Forest': RandomForestRegressor(n_estimators=100, random_state=42),
        'Gradient Boosting': GradientBoostingRegressor(n_estimators=100, random_state=42)
    }
    
    # 训练与评估
    results = {}
    best_model = None
    best_r2 = -float('inf')
    
    with st.spinner("模型训练中..."):
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
                'MAE': mae, 'MSE': mse, 'RMSE': rmse, 'R2': r2
            }
            
            if r2 > best_r2:
                best_r2 = r2
                best_model = pipeline
    
    # 展示结果
    st.success("模型训练完成！")
    
    st.subheader("各模型性能对比")
    comparison_df = pd.DataFrame(results).T[['MAE', 'RMSE', 'R2']]
    st.dataframe(comparison_df, use_container_width=True)
    
    # R2 对比图
    st.subheader("模型 R² 分数对比")
    fig, ax = plt.subplots(figsize=(10, 5))
    comparison_df['R2'].plot(kind='bar', color=['#1f77b4', '#ff7f0e', '#2ca02c'], ax=ax)
    ax.set_title('各模型 R² 分数对比')
    ax.set_ylabel('R² 分数')
    ax.set_ylim(0, 1)
    st.pyplot(fig)
    
    # 保存最优模型
    with open('movie_revenue_model.pkl', 'wb') as f:
        pickle.dump(best_model, f)
    
    st.info(f"✅ 最优模型：{[k for k, v in models.items() if best_model.named_steps['regressor'] is v][0]}")
    st.info(f"✅ 最优 R² 分数：{best_r2:.4f}")
    st.success("模型已保存为 movie_revenue_model.pkl")

# --------------------------
# 4. 票房预测
# --------------------------
elif menu == "票房预测":
    st.subheader("🎯 实时票房预测")
    
    # 加载模型
    try:
        with open('movie_revenue_model.pkl', 'rb') as f:
            model = pickle.load(f)
    except:
        st.error("请先在【模型训练与评估】页面训练并保存模型！")
        st.stop()
    
    # 输入表单
    with st.form("movie_form"):
        col1, col2 = st.columns(2)
        
        with col1:
            budget = st.number_input("预算（美元）", min_value=1000000, max_value=500000000, value=150000000, step=1000000)
            popularity = st.number_input("热度指数", min_value=1.0, max_value=100.0, value=85.5)
            vote_avg = st.number_input("评分", min_value=4.0, max_value=9.5, value=8.2)
            vote_count = st.number_input("投票人数", min_value=100, max_value=200000, value=50000)
            
        with col2:
            runtime = st.number_input("时长（分钟）", min_value=80, max_value=200, value=135)
            language = st.selectbox("语言", ['en', 'zh', 'ja', 'ko', 'fr', 'es', 'de'])
            genre = st.selectbox("类型", ['Action', 'Adventure', 'Animation', 'Comedy', 'Crime', 'Documentary', 'Drama', 'Fantasy', 'Horror', 'Romance', 'Science Fiction', 'Thriller'])
            company = st.selectbox("出品公司", ['Disney', 'Warner Bros', 'Universal', 'Paramount', 'Sony', 'Fox', 'Netflix', 'Amazon', 'MGM', 'Lionsgate'])
        
        year = st.slider("上映年份", 2010, 2030, 2024)
        month = st.slider("上映月份", 1, 12, 7)
        
        submit = st.form_submit_button("开始预测票房")
    
    # 预测
    if submit:
        sample = pd.DataFrame({
            'Budget': [budget],
            'Popularity': [popularity],
            'Vote_Average': [vote_avg],
            'Vote_Count': [vote_count],
            'Original_Language': [language],
            'Runtime': [runtime],
            'Genres': [genre],
            'Production_Companies': [company],
            'Release_Year': [year],
            'Release_Month': [month]
        })
        
        with st.spinner("预测中..."):
            pred = model.predict(sample)
        
        st.success("预测完成！")
        st.metric("预测票房收入", f"${pred[0]:,.2f}")
        st.metric("预测票房（百万美元）", f"{pred[0]/1e6:.2f} M")

# --------------------------
# 页脚
# --------------------------
st.markdown("---")
st.markdown("🎬 电影票房预测系统 | 基于 Streamlit + 机器学习")
