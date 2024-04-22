import pandas as pd
from sklearn.model_selection import train_test_split

# 读取 CSV 文件
csv_file_path = '/home/zhangmin/toby/IBA_Project_24spr/data/flight_review/flight_review.csv'
df = pd.read_csv(csv_file_path, quotechar='"', escapechar='\\', engine='python', on_bad_lines='skip')

# 选择相关列并重命名
df = df[['text', 'airline_sentiment']]
df.rename(columns={'text': 'input', 'airline_sentiment': 'output'}, inplace=True)

# 替换输出值
df['output'] = df['output'].replace({
    'negative': 'A. negative',
    'neutral': 'B. neutral',
    'positive': 'C. positive'
})

# 添加指令和历史字段
df['instruction'] = 'What is the sentiment of this tweet? Please choose an answer from {A. negative; B. neutral; C. positive}.'
df['history'] = [[] for _ in range(len(df))]

# 选择最终列的顺序
df = df[['instruction', 'input', 'output', 'history']]

# 打乱 DataFrame
df = df.sample(frac=1).reset_index(drop=True)

# 分割为训练集和测试集，比例为 0.8:0.2
train_df, test_df = train_test_split(df, test_size=0.2, random_state=42)

# 保存为 JSON 文件
train_json_file_path = '/home/zhangmin/toby/IBA_Project_24spr/data/flight_review/train.json'
test_json_file_path = '/home/zhangmin/toby/IBA_Project_24spr/data/flight_review/test.json'

train_df.to_json(train_json_file_path, orient='records', force_ascii=False, lines=True, indent=2)
test_df.to_json(test_json_file_path, orient='records', force_ascii=False, lines=True, indent=2)
