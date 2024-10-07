import os
import csv
import json
import argparse
import numpy as np
from tqdm import tqdm
from joblib import dump, load
from sklearn.model_selection import train_test_split
from xgboost import XGBClassifier
from sklearn.metrics import accuracy_score
from lightgbm import LGBMClassifier
from catboost import CatBoostClassifier
from sklearn.pipeline import Pipeline
from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import cross_val_score

'''
├── 2018
|   ├── ...txt
|   └── ...txt
└── checkpoints
|   ├── AmpRcRllRn
|   └── AmoxRcRllRn-ClavulanRc acRd
|   └── ...
├── test.py
└── 2018_clean.csv
└── checkpoints
└── metrics.json
'''

folder_path = "./2018" 
csv_path = "2018_clean.csv"
RATIO_OF_TEST = 0.1

reaction_to_number = {"NaN": 0, "S": 0, "R": 1}
number_to_reaction = {0: "non_R", 1: "R"}
NUM_OF_REACTION = 2

filter_num = 1000

X = []
labels = []
each_antibiotic_name = []
each_labels = []
valid_antibiotic = []
metrics_saver_cv = {"lightgbm": {}, "catboost": {}, "xgboost": {}} #cross-validation
metrics_saver = {"lightgbm": {}, "catboost": {}, "xgboost": {}}

#前處理: 把對應的菌種特徵以及抗生素的label讀取近來，並且過濾掉R類別個數 <= 1000的抗生素類別
def preprocessing_X():
    global each_antibiotic_name, each_labels, valid_antibiotic
    with open(csv_path, newline='') as csvfile:
        rows = list(csv.reader(csvfile))
        each_antibiotic_name = rows[0][1:]
        each_labels = [[] for _ in range(len(each_antibiotic_name))]
        valid_antibiotic = [True for _ in range(len(each_antibiotic_name))]
        for row in tqdm(rows[1:], total=len(rows[1:]), leave=False):#用 tqdm 迭代CSV文件中除第一行外的每一行
            file_name = row[0]
            feature_list = []
            target_path = folder_path + "/" + file_name + ".txt"

            if not os.path.exists(target_path): #沒找到該檔名的特徵就跳過
                print(target_path," file not exists !")
                continue
            with open(target_path, 'r') as file:
                next(file) #跳過 bin_index binned_intensity 這行
                for line in file:
                    parts = line.split() # 分割每一行的數據，以空格為分隔符
                    feature_list.append(float(parts[1])) # 取得 binned_intensity 的數值（第二個元素），轉換成浮點數後加入特徵中

            X.append(feature_list)  #每個樣本的特徵加到x
            for i, reaction in enumerate(row[1:]):
                each_labels[i].append(reaction_to_number[reaction])

    #為了避免 R S NaN 沒有一起出現在訓練集, 譬如AmoxRcRllRn整欄都是NaN(這樣是不允許的), valid_antibiotic 用來判斷該抗生素是否該被訓練
    cnt = {} #存反應類型
    for i in range(len(each_antibiotic_name)):
        cnt[each_antibiotic_name[i]] = [ 0 for _ in range(NUM_OF_REACTION)]
        reaction_occur = [ False for _ in range(NUM_OF_REACTION)]
        for reaction in each_labels[i]:#統計反應類型
            reaction_occur[reaction] = True
            cnt[each_antibiotic_name[i]][reaction] += 1 #反應類型出現次數
        for occur in reaction_occur:
            if not occur or cnt[each_antibiotic_name[i]][1] < filter_num: #少於1000就不訓練
                valid_antibiotic[i] = False
    with open("./count.json", "w", encoding="utf8") as outfile:
        json.dump(cnt, outfile)

def every_antibiotic(): #對有效的抗生素進行訓練
    print("Start training!")
    for i, antibiotic in tqdm(enumerate(each_antibiotic_name), total=len(each_antibiotic_name), leave=False):
        if valid_antibiotic[i]:
            print(f"第{i+1}個抗生素:",end='')
            training_and_testing(i,antibiotic)
        else:
            print(f"第{i+1}個抗生素沒有包含所有可能:",antibiotic)

def get_pipeline_and_parameters(model, random_state):
    if model == 'lightgbm':
        lightgbm = LGBMClassifier(random_state=random_state, n_jobs=4, device='gpu', verbosity=-1)
        pipeline = Pipeline(steps=[('lightgbm', lightgbm)])
        param_grid = {
            'lightgbm__n_estimators': [100, 200],
            'lightgbm__learning_rate': [0.01, 0.1]
        }
        return pipeline, param_grid

    elif model == 'catboost':
        catboost = CatBoostClassifier(random_state=random_state, silent=True, task_type='GPU')
        pipeline = Pipeline(steps=[('catboost', catboost)])
        param_grid = {
            'catboost__iterations': [50, 100, 200],                
            'catboost__learning_rate': [0.01, 0.1, 0.2],          
            'catboost__max_depth': [3, 5, 7],                          
            'catboost__l2_leaf_reg': [1, 1.5, 3]
        }
        return pipeline, param_grid
    
    elif model == 'xgboost':
        xgboost = XGBClassifier(use_label_encoder=False, eval_metric='logloss', random_state=random_state)
        pipeline = Pipeline(steps=[('xgboost', xgboost)])
        param_grid = {
            'xgboost__n_estimators': [100, 200],
            'xgboost__min_child_weight': [1, 3, 5],
            'xgboost__gamma': [0, 0.05, 0.3, 0.5, 0.7, 1.0],
            'xgboost__subsample': [0.6, 0.8, 1.0],
            'xgboost__colsample_bytree': [0.6, 0.8, 1.0],
            'xgboost__max_depth': [3, 5, 6]
        }
        return pipeline, param_grid
    else:
        raise ValueError(f'Invalid model name: {model}')

def training_and_testing(index, antibiotic_name, random_state=2024):
    label = each_labels[index]
    X_train, X_test, Y_train, Y_test = train_test_split(X, label, test_size=RATIO_OF_TEST, random_state=random_state, stratify=label)

    K_folds = 3
    model_names = ['lightgbm', 'catboost', 'xgboost']
    # model_names = ['catboost']
    # avg_cv_scores = []
    scores = [] #存個別模型的分數

    save_model_path = os.path.join("./checkpoints_new", antibiotic_name) #把model存在./checkpoints_new
    os.makedirs(save_model_path, exist_ok=True) #確保目錄已經存在
    for model_name in tqdm(model_names, total=len(model_names), leave=False):
        pipeline, param_grid = get_pipeline_and_parameters(model_name, random_state)
        grid_search = GridSearchCV(pipeline, param_grid, cv=K_folds, scoring='accuracy')
        grid_search.fit(X_train, Y_train) #找出最佳參數
        best_model = grid_search.best_estimator_
        predictions = best_model.predict(X_test)
        accuracy = accuracy_score(Y_test, predictions)
        scores.append(accuracy)
        dump(best_model, os.path.join(save_model_path, f"{model_name}.joblib"))
        #透過joblib的dump把model保存到localhost
        # cv_scores = cross_val_score(pipeline, X_train, Y_train, cv=K_folds, scoring='accuracy', n_jobs=4)
        # mean_cv_score = np.mean(cv_scores)
        # avg_cv_scores.append(mean_cv_score)
        
    for model_name, score in zip(model_names, scores): #zip起來
        metrics_saver[model_name][antibiotic_name] = score
    with open("./metrics.json", "w", encoding="utf8") as outfile:
        json.dump(metrics_saver, outfile) #把model_names, antibiotic_name],scores 寫進metrics.json

def inference(args, random_state=2024): #預測模型使用預訓練的模型
    save_model_path = os.path.join("./checkpoints", args.antibiotic_name)
    model_path = os.path.join(save_model_path, f"{args.model_name}.joblib")
    model = load(model_path)
    print(f"Model loaded from {model_path}")
    
    if args.mode == "test":
        target_path = folder_path + "/" + args.strain_name + ".txt"
        feature_list = []
        with open(target_path, 'r') as file:
            next(file) #跳過 bin_index binned_intensity 這行
            for line in file:
                parts = line.split() # 分割每一行的數據，以空格為分隔符
                feature_list.append(float(parts[1])) # 取得 binned_intensity 的數值（第二個元素），轉換成浮點數後加入特徵中
        feature_list = np.array(feature_list).reshape(1, -1)
        prediction = model.predict(feature_list)[0]
        prediction = number_to_reaction[prediction]
        print(f"Prediction for antibiotic {args.antibiotic_name}: {prediction}")
    elif args.mode == "eval":
        preprocessing_X()
        label = each_labels[each_antibiotic_name.index(args.antibiotic_name)]
        _, X_test, _, Y_test = train_test_split(X, label, test_size=RATIO_OF_TEST, random_state=random_state, stratify=label)
        prediction = model.predict(X_test)
        accuracy = accuracy_score(Y_test, prediction)
        print(f"{args.antibiotic_name} Accuracy on model {args.model_name}: {accuracy}")
    else:
        raise ValueError(f'Invalid mode: {args.mode}')

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--mode', type=str, default='eval', help='Training or testing mode (train/eval/test)') #可以改成train或eval(已經訓練好的模型)
    
    # If mode == 'test' or 'eval' 設定
    parser.add_argument('--model_name', type=str, default='xgboost', help='Select model (lightgbm/catboost/xgboost)')
    parser.add_argument('--antibiotic_name', type=str, default='AmpRcRllRn', help='Select antibiotic name')
    parser.add_argument('--strain_name', type=str, default='725696ba-1be2-4130-b357-400037987f5c_3312', help='Select strain name')
    args = parser.parse_args()
    
    if args.mode == "train":
        preprocessing_X()
        every_antibiotic()
        # with open("./metrics_cv.json", "w", encoding="utf8") as outfile:
        #     json.dump(metrics_saver_cv, outfile)
        with open("./metrics.json", "w", encoding="utf8") as outfile:
            json.dump(metrics_saver, outfile)
    elif args.mode == "test" or args.mode == "eval":
        inference(args)
    else:
        raise ValueError(f'Invalid mode: {args.mode}')
