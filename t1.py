import os
import numpy as np
import csv
from sklearn.model_selection import train_test_split
from xgboost import XGBClassifier
from sklearn.metrics import accuracy_score
from lightgbm import LGBMClassifier
from catboost import CatBoostClassifier
from xgboost import XGBClassifier
from sklearn.pipeline import Pipeline



'''
檔案格式, 相對路徑

├── 2018
|   ├── ...txt
|   └── ...txt
├── test.py
└── 2018_clean.csv
'''

folder_path = "./2018" # 自行定義的檔案夾路徑
csv_path = "2018_clean.csv"
RATIO_OF_TEST = 0.1 #定義testing set的size要切出多少 (最大為1)
reaction_to_number = {"NaN": 0, "S": 1, "R": 2, }
NUM_OF_REACTION = 3


X = []
labels = []
each_antibiotic_name = []
each_labels = []
valid_antibiotic = []

def preprocessing_X():
    global each_antibiotic_name, each_labels, valid_antibiotic
    with open(csv_path, newline='') as csvfile:
        rows = list(csv.reader(csvfile))
        each_antibiotic_name = rows[0][1:]
        each_labels = [[] for _ in range(len(each_antibiotic_name))]
        valid_antibiotic = [True for _ in range(len(each_antibiotic_name))]
        for row in rows[1:]:
            file_name = row[0]
            feature_list = []
            target_path = folder_path + "/" + file_name + ".txt"

            if not os.path.exists(target_path): #csv沒找到該檔名就跳過
                print(target_path," file not exists !")
                continue
            with open(target_path, 'r') as file:
                next(file) #跳過 bin_index binned_intensity 這行
                for line in file:
                    parts = line.split() # 分割每一行的數據，以空格為分隔符
                    feature_list.append(float(parts[1])) # 取得 binned_intensity 的數值（第二個元素），轉換成浮點數後加入特徵中

            X.append(feature_list)  #每個樣本的特徵
            for i,reaction in enumerate(row[1:]):
                each_labels[i].append(reaction_to_number[reaction])

    #以下是為了避免 R S NaN 沒有一起出現在訓練集, 譬如AmoxRcRllRn整欄都是NaN(這樣是不允許的), valid_antibiotic 用來判斷該抗生素是否該被訓練
    for i in range(len(each_antibiotic_name)):
        reaction_occur = [ False for _ in range(NUM_OF_REACTION)]
        for reaction in each_labels[i]:
            reaction_occur[reaction] = True
        for occur in reaction_occur:
            if not occur:
                valid_antibiotic[i] = False


def every_antibiotic():
    for i, antibiotic in enumerate(each_antibiotic_name):
        if valid_antibiotic[i]:
            print(f"第{i+1}個抗生素:",end='')
            training_and_testing(i,antibiotic)
        else:
            print(f"第{i+1}個抗生素沒有包含所有可能:",antibiotic)

def training_and_testing(index, antibiotic_name):
    label = each_labels[index]

    X_train, X_test, Y_train, Y_test = train_test_split(X, label, test_size=RATIO_OF_TEST) 
    model = XGBClassifier(objective='multi:softmax', use_label_encoder=False, num_class=len(set(label)))
    model.fit(np.array(X_train), np.array(Y_train)) # 訓練模型
    predictions = model.predict(np.array(X_test)) # 預測

    accuracy = accuracy_score(Y_test, predictions) # 計算準確性
    print(f'{antibiotic_name}的預測準確率: {accuracy:.2f}')


if __name__ == '__main__':
    preprocessing_X()
    every_antibiotic()

