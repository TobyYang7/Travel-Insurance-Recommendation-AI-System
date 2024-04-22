import pandas as pd
from sklearn.model_selection import train_test_split


def format(df):
    # 选择需要的列并重命名
    df = df[['text', 'airline_sentiment']].copy()
    df.rename(columns={'text': 'input', 'airline_sentiment': 'output'}, inplace=True)

    # 替换output列的值
    df.loc[:, 'output'] = df['output'].replace({
        'negative': 'A. negative',
        'neutral': 'B. neutral',
        'positive': 'C. positive'
    })

    # 添加instruction列
    df.loc[:, 'instruction'] = 'What is the sentiment of this tweet? Please choose an answer from {A. negative; B. neutral; C. positive}.'

    # 添加history列
    df.loc[:, 'history'] = pd.Series([[] for _ in range(len(df))], index=df.index)

    # 重新排序列
    df = df[['instruction', 'input', 'output', 'history']]
    return df


csv_file_path = '/home/zhangmin/toby/IBA_Project_24spr/data/flight_review/Tweets.csv'
df = pd.read_csv(csv_file_path, quotechar='"', escapechar='\\', engine='python', on_bad_lines='skip')
train_df, test_df = train_test_split(df, test_size=0.2, random_state=42)

train_df = format(train_df)[:3000]
test_df = format(test_df)

train_json_path = '/home/zhangmin/toby/IBA_Project_24spr/data/flight_review/train.json'
test_json_path = '/home/zhangmin/toby/IBA_Project_24spr/data/flight_review/test.json'
train_df.to_json(train_json_path, orient='records', force_ascii=False, lines=False, indent=2)
test_df.to_json(test_json_path, orient='records', force_ascii=False, lines=False, indent=2)
