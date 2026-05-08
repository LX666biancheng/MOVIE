import pandas as pd
import numpy as np
import ast
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MultiLabelBinarizer

# 1. 读取数据
movies = pd.read_csv(r"C:\Users\曹中洋\Downloads\tmdb_5000_movies.csv")
credits = pd.read_csv(r"C:\Users\曹中洋\Downloads\tmdb_5000_movies.csv")

# 2. 合并数据
df = movies.merge(
    credits,
    left_on="id",
    right_on="movie_id",
    how="inner",
    suffixes=("", "_credits")
)

# 3. 字段理解
# 目标变量：revenue，表示电影票房收入
# 输入特征：budget, popularity, vote_average, vote_count,
# original_language, runtime, genres, release_date,
# production_companies, cast, crew 等

# 4. JSON 字段解析函数
def parse_json_names(x):
    try:
        data = ast.literal_eval(x)
        return [item["name"] for item in data]
    except:
        return []

def get_director(x):
    try:
        data = ast.literal_eval(x)
        for item in data:
            if item.get("job") == "Director":
                return item.get("name")
        return "Unknown"
    except:
        return "Unknown"

def get_top_cast(x, n=3):
    try:
        data = ast.literal_eval(x)
        return [item["name"] for item in data[:n]]
    except:
        return []

# 5. 解析复杂字段
df["genres_list"] = df["genres"].apply(parse_json_names)
df["production_companies_list"] = df["production_companies"].apply(parse_json_names)
df["cast_list"] = df["cast"].apply(get_top_cast)
df["director"] = df["crew"].apply(get_director)

# 6. 缺失值处理
df["runtime"] = df["runtime"].fillna(df["runtime"].median())
df["overview"] = df["overview"].fillna("")
df["tagline"] = df["tagline"].fillna("Unknown")
df["homepage"] = df["homepage"].fillna("No Homepage")
df["release_date"] = pd.to_datetime(df["release_date"], errors="coerce")

# 删除没有目标变量或日期严重缺失的数据
df = df.dropna(subset=["release_date", "revenue"])

# 7. 异常值处理
# budget 和 revenue 中的 0 通常表示未知，转为缺失
df["budget"] = df["budget"].replace(0, np.nan)
df["revenue"] = df["revenue"].replace(0, np.nan)

# 删除 revenue 缺失的数据，因为 revenue 是预测目标
df = df.dropna(subset=["revenue"])

# budget 用中位数填补
df["budget"] = df["budget"].fillna(df["budget"].median())

# 片长异常处理：保留 30 到 300 分钟之间的电影
df = df[(df["runtime"] >= 30) & (df["runtime"] <= 300)]

# 评分异常处理：评分应在 0 到 10 之间
df = df[(df["vote_average"] >= 0) & (df["vote_average"] <= 10)]

# 8. 日期特征转换
df["release_year"] = df["release_date"].dt.year
df["release_month"] = df["release_date"].dt.month
df["release_quarter"] = df["release_date"].dt.quarter
df["release_weekday"] = df["release_date"].dt.weekday

# 9. 特征工程
df["genres_count"] = df["genres_list"].apply(len)
df["production_company_count"] = df["production_companies_list"].apply(len)
df["cast_count"] = df["cast_list"].apply(len)

df["is_english"] = (df["original_language"] == "en").astype(int)

# 偏态变量对数变换
df["log_budget"] = np.log1p(df["budget"])
df["log_revenue"] = np.log1p(df["revenue"])
df["log_popularity"] = np.log1p(df["popularity"])
df["log_vote_count"] = np.log1p(df["vote_count"])

# 10. 类型 one-hot 编码
mlb = MultiLabelBinarizer()
genre_encoded = pd.DataFrame(
    mlb.fit_transform(df["genres_list"]),
    columns=["genre_" + g for g in mlb.classes_],
    index=df.index
)

df = pd.concat([df, genre_encoded], axis=1)

# 11. 语言类别编码
language_encoded = pd.get_dummies(
    df["original_language"],
    prefix="lang",
    drop_first=True
)

df = pd.concat([df, language_encoded], axis=1)

# 12. 导演类别编码
# 高频导演保留，低频归为 Other
top_directors = df["director"].value_counts().head(20).index
df["director_group"] = df["director"].apply(
    lambda x: x if x in top_directors else "Other"
)

director_encoded = pd.get_dummies(
    df["director_group"],
    prefix="director",
    drop_first=True
)

df = pd.concat([df, director_encoded], axis=1)

# 13. 选择建模字段
feature_cols = [
    "log_budget",
    "log_popularity",
    "vote_average",
    "log_vote_count",
    "runtime",
    "release_year",
    "release_month",
    "release_quarter",
    "release_weekday",
    "genres_count",
    "production_company_count",
    "cast_count",
    "is_english"
]

feature_cols += list(genre_encoded.columns)
feature_cols += list(language_encoded.columns)
feature_cols += list(director_encoded.columns)

X = df[feature_cols]
y = df["log_revenue"]

# 14. 数据划分
# 训练集 70%，验证集 15%，测试集 15%
X_train, X_temp, y_train, y_temp = train_test_split(
    X,
    y,
    test_size=0.30,
    random_state=42
)

X_val, X_test, y_val, y_test = train_test_split(
    X_temp,
    y_temp,
    test_size=0.50,
    random_state=42
)

print("训练集：", X_train.shape)
print("验证集：", X_val.shape)
print("测试集：", X_test.shape)

# 15. 保存处理后的数据
train_data = X_train.copy()
train_data["log_revenue"] = y_train

val_data = X_val.copy()
val_data["log_revenue"] = y_val

test_data = X_test.copy()
test_data["log_revenue"] = y_test

train_data.to_csv("tmdb_train.csv", index=False)
val_data.to_csv("tmdb_validation.csv", index=False)
test_data.to_csv("tmdb_test.csv", index=False)

df.to_csv("tmdb_cleaned_full.csv", index=False)

print("数据预处理完成！")
