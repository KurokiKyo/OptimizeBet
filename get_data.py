from selenium import webdriver 
from selenium.webdriver.common.by import By
from webdriver_manager.chrome import ChromeDriverManager
from selenium.webdriver.chrome.options import Options
from time import sleep
import pandas as pd
from tqdm import tqdm

def import_data_from_JRA():
    print('Now Reading...')
    # for Windows --> browser = webdriver.Chrome(ChromeDriverManager().install())
    option = Options()
    option.add_argument('--headless')
    browser = webdriver.Chrome(options=option)

    url = 'https://www.jra.go.jp/'
    browser.get(url)
    xpath = '//*[@id="quick_menu"]/div/ul/li[3]/a'
    browser.find_element(by=By.XPATH,value=xpath).click() # トップページのオッズボタンをクリック

    youbi_num = 3
    kaijo_num = 5
    race_num = 12
    DATA = []
    for youbi in range(youbi_num):
        for kaijo in range(kaijo_num):
            try:
                xpath = '//*[@id="main"]/div[1]/div[{0}]/div/div/div[{1}]/a'.format(youbi+2, kaijo+1)
                browser.find_element(by=By.XPATH, value=xpath).click() # 今週のオッズの日程と開催場所の書いたボタンをクリック(ex. 3回新潟１日)
                date_place_xpath = '//*[@id="race_list"]/caption/div[1]/div/h2'
                date_and_place = browser.find_element(by=By.XPATH, value=date_place_xpath).text # 日時と開催場所のテキストを取得
            except Exception:
                break
            
            for race in range(race_num):
                try:
                
                    race_name_xpath = '//*[@id="race_list"]/tbody/tr[{}]/td[2]/div/div[1]'.format(race+1)
                    race_name = browser.find_element(by=By.XPATH, value=race_name_xpath).text # レース名を取得
                    race_xpath = '//*[@id="race_list"]/tbody/tr[{}]/th/a'.format(race+1)
                    browser.find_element(by=By.XPATH, value=race_xpath).click() # レースをクリック

                    odds_list = [] # 特定のレースにおける全ての馬のオッズを格納
                    for horse_idx in range(20):
                        idx_odds = [] # 特定の馬の馬番号, 名前, オッズを格納
                        for k in range(4):
                            try:
                                odds_xpath = '//*[@id="odds_list"]/table/tbody/tr[{0}]/td[{1}]'.format(horse_idx+1, k+1)
                                data = browser.find_element(by=By.XPATH, value=odds_xpath).text
                                if data != '' and len(idx_odds)<3:
                                    idx_odds.append(data)
                            except Exception as error:
                                #print(error)
                                break
                        if idx_odds==[]:break
                        else:odds_list.append(idx_odds)
                    
                    # odds_list はレースごとの['horse_idx', 'horse_name', 'tansho_odds']がレース数分連結された形式で格納
                    

                    DATA.append({'date_and_place':date_and_place, 'race_name':race_name, 'race_number':'{}R'.format(race+1), 'odds':odds_list})
                    #print(DATA)
                    
                    browser.back()
                except Exception:
                    break
            browser.back()
    DF = pd.DataFrame(DATA)
    DF.to_csv('keiba_data.csv')
    return DF


def get_name_and_odds(df, date, place, race_number):
    df_1 = df[df['date_and_place'].str.contains(place, na=False)]
    df_2 = df_1[df_1['date_and_place'].str.contains(date, na=False)]
    df_3 = df_2[df_2['race_number'].str.contains(str(race_number), na=False)]

    target = eval(pd.DataFrame(df_3['odds']).values.tolist()[0][0])

    idx_list = []
    name_list = []
    tansho_odds_list = []
    for data in target:
        idx_list.append(int(data[0]))
        name_list.append(data[1])
        try:
            odds = float(data[2])
        except Exception:
            odds = data[2]
        tansho_odds_list.append(odds)

    return name_list, tansho_odds_list




