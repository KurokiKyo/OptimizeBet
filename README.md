# OptimizeBet
Optimize bet for KEIBA

# Contents
get_data.py : JRAのサイトから、競馬のオッズ情報を自動で取得する機能のモジュール
keiba_generator.py : 競馬のオッズ情報から、勝率をHarvilleモデルで計算して上位x位までのランキングを生成してくれる機能と、そのランキングから、賭け方のいい組み合わせを自動的に提案してくれる機能のモジュール。

# Usage
詳細はtest.pyを参照

## JRAのWEBサイトからオッズのデータを読み込み, keiba_data.csvファイルを出力 注意）1〜2分かかる １度読み込むと2回目は不要
get_data.import_data_from_JRA() 
## get_data.pyで取得したオッズリストをデータフレームとして読み込む
df = pd.read_csv('keiba_data.csv')

## 日付、場所、レース番号を記述
date = '8月13日'
place = '札幌'
race_number = 5

## 取得したいレースの日時と場所と第何レースかを入力して馬の名前と単勝のオッズを取得
name_list, tansho_odds_list = get_data.get_name_and_odds(df=df, date=date, place=place, race_number=race_number) 

## インスタンスを作成
race = Keiba(tansho_odds_list=tansho_odds_list, race_name=date+place+str(race_number)+'R')

# ランキングを作成
race.calc_prob_ranking()
