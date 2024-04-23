import pandas as pd
from sklearn.model_selection import train_test_split
import datasets


def format_review(df):
    df = df[['text', 'airline_sentiment']].copy()
    df.rename(columns={'text': 'input', 'airline_sentiment': 'output'}, inplace=True)
    df.loc[:, 'output'] = df['output'].replace({
        'negative': 'A. negative',
        'neutral': 'B. neutral',
        'positive': 'C. positive'
    })
    df.loc[:, 'instruction'] = 'What is the sentiment of this tweet? Please choose an answer from {A. negative; B. neutral; C. positive}.'
    df.loc[:, 'history'] = pd.Series([[] for _ in range(len(df))], index=df.index)
    df = df[['instruction', 'input', 'output', 'history']]
    return df


csv_file_path = '../data/flight_review/Tweets.csv'
df = pd.read_csv(csv_file_path, quotechar='"', escapechar='\\', engine='python', on_bad_lines='skip')
train_df, test_df = train_test_split(df, test_size=0.2, random_state=42)

train_df = format_review(train_df)[:3000]
test_df = format_review(test_df)

train_json_path = '../data/flight_review/train.json'
test_json_path = '../data/flight_review/test.json'
train_df.to_json(train_json_path, orient='records', force_ascii=False, lines=False, indent=2)
test_df.to_json(test_json_path, orient='records', force_ascii=False, lines=False, indent=2)

insurance_ds = datasets.load_dataset('Ddream-ai/InsuranceCorpus')
insurance_df = pd.DataFrame(insurance_ds)
