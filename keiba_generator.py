import numpy as np
import pandas as pd
import math
from tqdm import tqdm
import requests
from bs4 import BeautifulSoup

class Keiba:
    def __init__(self, tansho_odds_list, race_name, refund_rate=0.8):
        self.num_horse = len(tansho_odds_list)
        self.tansho_odds_list = tansho_odds_list
        self.refund_rate = refund_rate
        self.first_prob = np.empty(0)
        self.second_prob = np.empty(0)
        self.third_prob = np.empty(0)
        self.win_prob_tansho_list = []
        self.win_prob_fukusho_list = []
        self.Pij = np.zeros((self.num_horse, self.num_horse))
        self.Pij_comb = np.zeros((self.num_horse, self.num_horse))
        self.Pij_wide = np.zeros((self.num_horse, self.num_horse))
        self.Pijk = np.zeros((self.num_horse, self.num_horse, self.num_horse))
        self.Pijk_comb = np.zeros((self.num_horse, self.num_horse, self.num_horse))
        self.tansho_bet = np.zeros(self.num_horse)
        self.fukusho_bet = np.zeros(self.num_horse)
        self.umaren_bet = np.zeros((self.num_horse, self.num_horse))
        self.umatan_bet = np.zeros((self.num_horse, self.num_horse))
        self.wide_bet = np.zeros((self.num_horse, self.num_horse))
        self.sanrenpuku_bet = np.zeros((self.num_horse, self.num_horse, self.num_horse))
        self.sanrentan_bet = np.zeros((self.num_horse, self.num_horse, self.num_horse))
        self.rank = []
        self.race_name = race_name

    def calc_win_prob_tansho(self):
        self.win_prob_tansho_list = []
        for odds in self.tansho_odds_list:
            self.win_prob_tansho_list.append(self.refund_rate/odds)    
    
    def calc_win_prob_fukusho(self):
        self.calc_win_prob_tansho()
        first_prob = np.array(self.win_prob_tansho_list)
        self.first_prob = first_prob
        secand_prob = np.zeros(len(self.win_prob_tansho_list))
        for k in range(len(self.win_prob_tansho_list)):
            tansho_prob = self.win_prob_tansho_list[k]
            prob = 0
            for i in range(len(self.win_prob_tansho_list)):
                if i != k:
                    prob += self.win_prob_tansho_list[i] * tansho_prob / (1 - self.win_prob_tansho_list[i])
            secand_prob[k] = prob
        self.second_prob = secand_prob
        third_prob = np.zeros(len(self.win_prob_tansho_list))
        for l in range(len(self.win_prob_tansho_list)):
            prob = 0
            for m in range(len(self.win_prob_tansho_list)):
                if m != l:
                    for n in range(len(self.win_prob_tansho_list)):
                        if n != m and n != l:
                            prob += self.win_prob_tansho_list[n] * self.win_prob_tansho_list[m] / (1 - self.win_prob_tansho_list[n]) * self.win_prob_tansho_list[l] / (1 - self.win_prob_tansho_list[m] - self.win_prob_tansho_list[n])
            third_prob[l] = prob
        self.third_prob = third_prob
        fukusho_prob = first_prob + secand_prob + third_prob
        self.win_prob_fukusho_list = fukusho_prob.tolist()

    def calc_win_prob_2rentan(self):
        self.calc_win_prob_tansho()
        for i in range(self.num_horse):
            for j in range(self.num_horse):
                if i == j:
                    self.Pij[i][j] = 0
                else:
                    self.Pij[i][j] = self.win_prob_tansho_list[i] * (self.win_prob_tansho_list[j] / (1 - self.win_prob_tansho_list[i]))
    
    def calc_win_prob_2renpuku(self):
        self.calc_win_prob_2rentan()
        self.Pij_comb = self.Pij + self.Pij.T
        self.Pij_comb = np.triu(self.Pij_comb)

    def calc_win_prob_3rentan(self):
        self.calc_win_prob_tansho()
        for i in range(self.num_horse):
            for j in range(self.num_horse):
                for k in range(self.num_horse):
                    if i == j or j == k or i == k:
                        self.Pijk[i][j][k] = 0
                    else:
                        self.Pijk[i][j][k] = self.win_prob_tansho_list[i] * (self.win_prob_tansho_list[j] / (1 - self.win_prob_tansho_list[i])) * (self.win_prob_tansho_list[k] / (1 - self.win_prob_tansho_list[i] - self.win_prob_tansho_list[j]))


    def calc_win_prob_3renpuku(self):
        self.calc_win_prob_3rentan()
        for i in range(self.num_horse):
            for j in range(self.num_horse):
                for k in range(self.num_horse):
                    if i < j and j < k :
                        self.Pijk_comb[i][j][k] = self.Pijk[i][j][k] + self.Pijk[j][i][k] + self.Pijk[j][k][i] + self.Pijk[i][k][j] + self.Pijk[k][i][j] + self.Pijk[k][j][i]
                        self.Pijk_comb[i][k][j] = 0
                        self.Pijk_comb[j][i][k] = 0
                        self.Pijk_comb[j][k][i] = 0
                        self.Pijk_comb[k][i][j] = 0
                        self.Pijk_comb[k][j][i] = 0
                        # self.Pijk_comb[i][k][j] = self.Pijk_comb[i][j][k]
                        # self.Pijk_comb[j][i][k] = self.Pijk_comb[i][j][k]
                        # self.Pijk_comb[j][k][i] = self.Pijk_comb[i][j][k]
                        # self.Pijk_comb[k][i][j] = self.Pijk_comb[i][j][k]
                        # self.Pijk_comb[k][j][i] = self.Pijk_comb[i][j][k]

    def calc_win_prob_wide(self):
        self.calc_win_prob_2rentan()
        self.calc_win_prob_3rentan()
        Pij_13 = np.zeros((self.num_horse, self.num_horse))
        Pij_23 = np.zeros((self.num_horse, self.num_horse))
        for i in range(self.num_horse):
            for j in range(self.num_horse):
                for k in range(self.num_horse):
                    if not (i == j or j == k or i == k):
                        Pij_13[i][j] += self.Pijk[i][k][j]
                        Pij_23[i][j] += self.Pijk[k][i][j]
        self.Pij_wide = self.Pij + Pij_13 + Pij_23 + self.Pij.T + Pij_13.T + Pij_23.T
        self.Pij_wide = np.triu(self.Pij_wide)

    
    def calc_win_prob_all(self, criteria=0.01):
        self.calc_win_prob_tansho()
        self.calc_win_prob_fukusho()
        self.calc_win_prob_2rentan()
        self.calc_win_prob_2renpuku()
        self.calc_win_prob_wide()
        self.calc_win_prob_3rentan()
        self.calc_win_prob_3renpuku()

        count = np.count_nonzero(np.array(self.win_prob_tansho_list)>criteria) + np.count_nonzero(np.array(self.win_prob_fukusho_list)>criteria) + np.count_nonzero(np.round(self.Pij,5)>criteria) + np.count_nonzero(np.round(self.Pij_comb,5)>criteria) + np.count_nonzero(np.round(self.Pij_wide,5)>criteria) + np.count_nonzero(np.round(self.Pijk,5)>criteria) + np.count_nonzero(np.round(self.Pijk_comb,5)>criteria)
        print('{0}%以上の的中確率がある馬券は{1}種類です'.format(criteria*100, count))
    
    def output_prob_all(self):
        df_tansho = pd.DataFrame(100*np.round(np.array(self.win_prob_tansho_list),5))
        df_fukusho = pd.DataFrame(100*np.round(np.array(self.win_prob_fukusho_list),5))
        df_single = pd.concat({"tansho":df_tansho, "fukusho":df_fukusho})
        df_single.to_csv("prob_single.csv")

        df_umatan = pd.DataFrame(100*np.round(self.Pij,5))
        df_umaren = pd.DataFrame(100*np.round(self.Pij_comb,5))
        df_wide = pd.DataFrame(100*np.round(self.Pij_wide,5))
        df_double = pd.concat({"umatan":df_umatan, "umaren":df_umaren, "wide":df_wide})
        df_double.to_csv("prob_double.csv")
        for n in range(self.num_horse):
            if n == 0:
                df_3rentan = pd.DataFrame(100*np.round(self.Pijk[:,:,n], 5))
            else:
                df_3rentan = pd.concat([df_3rentan, pd.DataFrame(100*np.round(self.Pijk[:,:,n], 5))])
        for m in range(self.num_horse):
            if m == 0:
                df_3renpuku = pd.DataFrame(100*np.round(self.Pijk_comb[:,:,m],5))
            else:
                df_3renpuku = pd.concat([df_3renpuku, pd.DataFrame(100*np.round(self.Pijk_comb[:,:,m],5))])
        df_triple = pd.concat({"3rentan":df_3rentan, "3renpuku":df_3renpuku})
        df_triple.to_csv("prob_triple.csv")
    
    def calc_prob_ranking(self, num_rank=200, criteria=0.01, coeff=3, max_amount=5000, risk_hedge=False, tansho=True, fukusho=True, umaren=True, umatan=True, wide=True, sanrenpuku=True, sanrentan=True):
        if not (tansho or fukusho or umaren or umatan or wide or sanrenpuku or sanrentan):
            print("少なくとも１つは賭け方を選択してください")
            return
        if max_amount<1000:
            print('予算額を1000円以上に設定してください')
            return
        self.calc_win_prob_all(criteria=criteria)
        if risk_hedge:
            minbet = 100
        else:
            minbet = 0
        # ['pattern', 'horse', 'probability','bet']の順でリストに格納
        rank_list = []
        tansho_prob = np.array(self.win_prob_tansho_list)
        fukusho_prob = np.array(self.win_prob_fukusho_list)
        tansho_prob_copy = np.copy(tansho_prob)
        fukusho_prob_copy = np.copy(fukusho_prob)
        Pij_copy = np.copy(self.Pij)
        Pij_comb_copy = np.copy(self.Pij_comb)
        Pij_wide_copy = np.copy(self.Pij_wide)
        Pijk_copy = np.copy(self.Pijk)
        Pijk_comb_copy = np.copy(self.Pijk_comb)
        size = 0
        k = 0
        for option in [tansho, fukusho, umaren, umatan, wide, sanrenpuku, sanrentan]:
            if option and (k==0 or k==1):
                size += self.num_horse
            elif option and (k==2 or k==4):
                size += int(self.num_horse*(self.num_horse-1)*0.5)
            elif option and k==3:
                size += self.num_horse*(self.num_horse-1)
            elif option and k==5:
                size += int(self.num_horse*(self.num_horse-1)*(self.num_horse-2)/6)
            elif option and k==6:
                size += self.num_horse*(self.num_horse-1)*(self.num_horse-2)
            k+=1

        #while len(rank_list) < num_rank:
        for i in tqdm(range(num_rank)):
            if len(rank_list) == size:
                print('※ランキング数が全種類の馬券数に到達しました')
                break
            index = np.argmax(np.array([np.max(tansho_prob),np.max(fukusho_prob),np.max(self.Pij), np.max(self.Pij_comb), np.max(self.Pij_wide), np.max(self.Pijk), np.max(self.Pijk_comb)]))
            if index == 0:
                argmax = np.argmax(tansho_prob)
                if tansho:
                    x = self.calc_bet(criteria=criteria, bet_array=self.tansho_bet, prob_array=tansho_prob_copy, index_list=[argmax],coeff=coeff)
                    rank_list.append(['tansho', [argmax+1], np.max(tansho_prob), x])
                tansho_prob[argmax] = 0
            elif index == 1:
                argmax = np.argmax(fukusho_prob)
                if fukusho:
                    x = self.calc_bet(criteria=criteria, bet_array=self.fukusho_bet, prob_array=fukusho_prob_copy, index_list=[argmax], fukusho=True,coeff=coeff)
                    rank_list.append(['fukusho', [argmax+1], np.max(fukusho_prob),x])
                fukusho_prob[argmax] = 0
            elif index == 2:
                argmax = np.unravel_index(np.argmax(self.Pij), self.Pij.shape)
                if umatan:
                    x = self.calc_bet(criteria=criteria, bet_array=self.umatan_bet, prob_array=Pij_copy, index_list=np.array(argmax),coeff=coeff)
                    rank_list.append(['umatan', (np.array(argmax)+1).tolist(), np.max(self.Pij),x])
                self.Pij[argmax[0]][argmax[1]] = 0
            elif index == 3:
                argmax = np.unravel_index(np.argmax(self.Pij_comb), self.Pij_comb.shape)
                if umaren:
                    x = self.calc_bet(criteria=criteria, bet_array=self.umaren_bet, prob_array=Pij_comb_copy, index_list=np.array(argmax),coeff=coeff)
                    if x==0 and np.max(self.Pij_comb)>criteria:
                        rank_list.append(['wide', (np.array(argmax)+1).tolist(), np.max(self.Pij_wide), minbet])
                    else:    
                        rank_list.append(['umaren', (np.array(argmax)+1).tolist(), np.max(self.Pij_comb), x])
                self.Pij_comb[argmax[0]][argmax[1]] = 0
            elif index == 4:
                argmax = np.unravel_index(np.argmax(self.Pij_wide), self.Pij_wide.shape)
                if wide:
                    x = self.calc_bet(criteria=criteria, bet_array=self.wide_bet, prob_array=Pij_wide_copy, index_list=np.array(argmax), wide=True,coeff=coeff)
                    if x==0 and np.max(self.Pij_wide)>criteria:
                        rank_list.append(['wide', (np.array(argmax)+1).tolist(), np.max(self.Pij_wide), minbet])
                    else:
                        rank_list.append(['wide', (np.array(argmax)+1).tolist(), np.max(self.Pij_wide), x])
                self.Pij_wide[argmax[0]][argmax[1]] = 0
            elif index == 5:
                argmax = np.unravel_index(np.argmax(self.Pijk), self.Pijk.shape)
                if sanrentan:
                    rank_list.append(['3rentan', (np.array(argmax)+1).tolist(), np.max(self.Pijk), self.calc_bet(criteria=criteria, bet_array=self.sanrentan_bet, prob_array=Pijk_copy, index_list=np.array(argmax),coeff=coeff)])
                self.Pijk[argmax[0]][argmax[1]][argmax[2]] = 0
            elif index == 6:
                argmax = np.unravel_index(np.argmax(self.Pijk_comb), self.Pijk_comb.shape)
                if sanrenpuku:
                    x = self.calc_bet(criteria=criteria, bet_array=self.sanrenpuku_bet, prob_array=Pijk_comb_copy, index_list=np.array(argmax),coeff=coeff)
                    if x==0 and np.max(self.Pijk_comb)>criteria:
                        rank_list.append(['3renpuku', (np.array(argmax)+1).tolist(), np.max(self.Pijk_comb), minbet])
                    else:
                        rank_list.append(['3renpuku', (np.array(argmax)+1).tolist(), np.max(self.Pijk_comb), x])
                self.Pijk_comb[argmax[0]][argmax[1]][argmax[2]] = 0
        self.rank = rank_list
        self.normalize_bet(max_amount)
        DF_rank = pd.DataFrame(self.rank, columns=['Pattern','Horse','Probability','Bet'])
        DF_rank.index = np.arange(1, len(DF_rank)+1)
        DF_rank.to_csv(str(self.race_name) +'.csv')
    
    def normalize_bet(self, max_amount):
        sum_bet = 0
        max_idx = 0
        for i in range(len(self.rank)):
            sum_bet += self.rank[i][3]
            if self.rank[max_idx][3]<self.rank[i][3]:
                max_idx = i
        x = 0
        for i in range(len(self.rank)):
            if sum_bet == 0: 
                break
            self.rank[i][3] = 100*round(self.rank[i][3]/sum_bet*max_amount/100)
            x += self.rank[i][3] 
        if x==0: 
            self.rank[max_idx][3] = 100
            print('良い組み合わせが見つかりませんでした。ベットタイプを変更するか、予算額を増やしてください')


    def calc_bet(self, criteria, bet_array, prob_array, index_list, fukusho=False, wide=False, coeff=1):
        #index_listの馬券にどれくらい賭けるかを決める関数
        dim = len(index_list)
        wide_bool = False
        fukusho_bool = False
        if wide and np.max(bet_array) != 0:
            bet_list = (np.nonzero(bet_array)[0].tolist())+(np.nonzero(bet_array)[1].tolist())
            wide_bool = ((index_list[0] in bet_list) or (index_list[1] in bet_list)) and len(np.nonzero(bet_array)[0])<3
        if fukusho and np.max(bet_array) != 0:
            fukusho_bool = np.count_nonzero(bet_array)<3 and prob_array[index_list[0]]>0.8
        if np.max(bet_array) == 0 or (fukusho and fukusho_bool) or (wide and wide_bool):
            
            if dim == 1:
                if prob_array[index_list[0]]<criteria: x=0
                else: x = 100*int(math.floor(1/criteria*prob_array[index_list[0]]**coeff))
                bet_array[index_list[0]] = x
            elif dim == 2:
                if prob_array[index_list[0]][index_list[1]]<criteria: x=0
                else : x = 100*int(math.floor(1/criteria*prob_array[index_list[0]][index_list[1]]**coeff))
                bet_array[index_list[0]][index_list[1]] = x
            elif dim == 3:
                if prob_array[index_list[0]][index_list[1]][index_list[2]]<criteria:x=0
                else : x = 100*int(math.floor(1/criteria*prob_array[index_list[0]][index_list[1]][index_list[2]]**coeff))
                bet_array[index_list[0]][index_list[1]][index_list[2]] = x
            return x
            
        else:
            if dim == 1:
                if prob_array[index_list[0]]<criteria:
                    x = 100
                else:
                    x = max(100*int(math.floor(1/criteria*prob_array[index_list[0]]**coeff)),100)
            elif dim == 2:
                if prob_array[index_list[0]][index_list[1]]<criteria:
                    x = 100
                else : x = max(100*int(math.floor(1/criteria*prob_array[index_list[0]][index_list[1]]**coeff)),100)
            elif dim == 3:
                if prob_array[index_list[0]][index_list[1]][index_list[2]]<criteria:
                    x = 0
                else: x = max(100*int(math.floor(1/criteria*prob_array[index_list[0]][index_list[1]][index_list[2]]**coeff)),100)
            else:
                print("error")
            
            #print(len(str(math.floor((1/criteria)*(criteria**(coeff))))))
            if coeff < 0 and x != 100 and len(str(math.floor((1/criteria)*(criteria**(coeff)))))>4: 
                x = np.round(10000*x/(10**len(str(math.ceil((1/criteria)*(criteria**(coeff)))))),-2)

            
            while x > 0:
                if dim == 1:
                    if np.sum(bet_array)/(0.8/prob_array[index_list[0]]-1) <= x and x <= np.nanmin((0.8/prob_array)*np.where(bet_array==0, 2*np.max(bet_array), bet_array))-np.sum(bet_array):
                        bet_array[index_list[0]] = x
                        break
                    else: x-=100
                elif dim == 2:
                    if np.sum(bet_array)/(0.8/prob_array[index_list[0]][index_list[1]]-1) <= x and x <= np.nanmin((0.8/prob_array)*np.where(bet_array==0, 2*np.max(bet_array), bet_array))-np.sum(bet_array):
                        bet_array[index_list[0]][index_list[1]] = x
                        break
                    else: x-=100
                elif dim == 3:
                    if np.sum(bet_array)/(0.8/prob_array[index_list[0]][index_list[1]][index_list[2]]-1) <= x and x <= np.nanmin((0.8/prob_array)*np.where(bet_array==0, 2*np.max(bet_array), bet_array))-np.sum(bet_array):
                        bet_array[index_list[0]][index_list[1]][index_list[2]] = x
                        break
                    else: x-=100
                else:
                    print('Error')
                    break
            return x
    
    def calc_return(self, result, tansho_odds, fukusho_odds, umaren_odds, umatan_odds, wide_odds, sanrenpuku_odds, sanrentan_odds):
        res_tansho = [result[0]]
        sort_res = sorted(result)
        res_fukusho = [[sort_res[0]],[sort_res[1]],[sort_res[2]]]
        res_umaren = [sort_res[0],sort_res[1]]
        res_umatan = [result[0],result[1]]
        res_wide = [[sort_res[0], sort_res[1]],[sort_res[0],sort_res[2]],[sort_res[1],sort_res[2]]]
        res_sanrenpuku = sort_res
        res_sanrentan = result
        sum_bet = 0
        sum_return = 0
        for bet_list in self.rank:
            if bet_list[0] == 'tansho' and (bet_list[1] == res_tansho):
                sum_return += tansho_odds*bet_list[3]
            elif bet_list[0] == 'fukusho' and (bet_list[1] in res_fukusho):
                sum_return += fukusho_odds[res_fukusho.index(bet_list[1])]*bet_list[3]
            elif bet_list[0] == 'umaren' and (bet_list[1] == res_umaren):
                sum_return += umaren_odds*bet_list[3]
            elif bet_list[0] == 'umatan' and (bet_list[1] == res_umatan):
                sum_return += umatan_odds*bet_list[3]
            elif bet_list[0] == 'wide' and (bet_list[1] in res_wide):
                sum_return += wide_odds[res_wide.index(bet_list[1])]*bet_list[3]
            elif bet_list[0] == '3renpuku' and (bet_list[1] == res_sanrenpuku):
                sum_return += sanrenpuku_odds*bet_list[3]
            elif bet_list[0] == '3rentan' and (bet_list[1] == res_sanrentan):
                sum_return += sanrentan_odds*bet_list[3]
            sum_bet += bet_list[3]
        if sum_bet==0:
            p = 100
        else:
            p = int(sum_return/sum_bet*100)
        print('掛け金:{0}円, リターン:{1}円, 回収率:{2}%, レース名:{3}:'.format(sum_bet,int(sum_return),p,self.race_name))
        return [sum_bet,int(sum_return),p,self.race_name]


    def get_odds_from_web(self, url):
        res = requests.get(url)
        soup = BeautifulSoup(res.text, 'html.parser')

    def calc_kitaichi(self, spin):
        kitaichi = 0
        for i in range(len(spin)):
            kitaichi += (self.win_prob_tansho_list[i]*self.tansho_odds_list[i]-1)*spin[i]
        return kitaichi
    
    def create_pQUBO_tansho(self, constraint=100, risk=2):
        self.calc_win_prob_tansho()
        num_invest = len(self.tansho_odds_list)
        h = np.zeros(len(self.tansho_odds_list))
        J = 2*constraint*np.ones((num_invest,num_invest))
        for i in range(len(self.tansho_odds_list)):
            h[i] = risk*(1-self.win_prob_tansho_list[i])-2*constraint + 1 - self.win_prob_tansho_list[i]*self.tansho_odds_list[i]
        return J , h
    
    def create_pQUBO_fukusho(self, constraint=1000, risk=2):
        self.calc_win_prob_fukusho()
        num_invest = len(self.tansho_odds_list)
        h = np.zeros(num_invest)
        J = 2*constraint*np.ones((num_invest,num_invest))
        for i in range(num_invest):
            h[i] = risk*(1-self.win_prob_fukusho_list[i])-2*constraint + 1 - self.win_prob_fukusho_list[i]*self.tansho_odds_list[i]
        return J , h
    
    def create_QUBO(self, prob_list, odds_list, constraint=100, risk_rate=2, capacity=10, amount=100):
        prob = np.array(prob_list)
        odds = np.array(odds_list)
        num_invest = len(odds_list)
        h = np.zeros(num_invest*capacity)
        J = np.zeros((num_invest*capacity,num_invest*capacity))
        for i in range(num_invest):
            idx = capacity*i
            for c in range(capacity):
                h[idx+c] = risk_rate*prob[i]*(odds[i]**2)-7*capacity*risk_rate*((prob[i]*odds[i])**2)-prob[i]*odds[i]+(1-2*amount)*constraint
        for i in range(num_invest):
            idx_i = capacity*i
            for j in range(num_invest):
                idx_j = capacity*j
                for c in range(capacity):
                    for d in range(capacity):
                        if not (idx_i+c == idx_j+d):
                            J[idx_i+c][idx_j+d] = 2*constraint - 14*risk_rate*capacity*prob[i]*prob[j]*odds[i]*odds[j]
        print(J)
        return J , h
        

