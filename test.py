from keiba_generator import Keiba 
import get_data
import pandas as pd

get_data.import_data_from_JRA() # JRAのWEBサイトからオッズのデータを読み込み, keiba_data.csvファイルを出力 注意）1〜2分かかる １度読み込むと2回目は不要

# --------引数設定---------
df = pd.read_csv('keiba_data.csv')
date = '8月13日'
place = '札幌'
race_number = 5

name_list, tansho_odds_list = get_data.get_name_and_odds(df=df, date=date, place=place, race_number=race_number) # 取得したいレースの日時と場所と第何レースかを入力して馬(ボートも追加予定)の名前と単勝のオッズを取得

race = Keiba(tansho_odds_list=tansho_odds_list, race_name=date+place+str(race_number)+'R')
race.calc_prob_ranking()
