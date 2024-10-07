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
from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import cross_val_score
'''
├── 2018
|   ├── ...txt
|   └── ...txt
├── test.py
└── 2018_clean.csv
'''
folder_path = "./2018" 
csv_path = "2018_clean.csv"
RATIO_OF_TEST = 0.1

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

    #為了避免 R S NaN 沒有一起出現在訓練集, 譬如AmoxRcRllRn整欄都是NaN(這樣是不允許的), valid_antibiotic 用來判斷該抗生素是否該被訓練
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

def get_pipeline_and_parameters(model, random_state):
    if model == 'lightgbm':
        lightgbm = LGBMClassifier(random_state=random_state)
        pipeline = Pipeline(steps=[('lightgbm', lightgbm)])
        param_grid = {
            'lightgbm__n_estimators': [100, 200],
            'lightgbm__learning_rate': [0.01, 0.1]
        }
        return pipeline, param_grid

    elif model == 'catboost':
        catboost = CatBoostClassifier(random_state=random_state, silent=True)
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
            'xgboost__min_child_weight': [1, 5, 10],
            'xgboost__gamma': [0.5, 1, 1.5, 2, 5],
            'xgboost__subsample': [0.6, 0.8, 1.0],
            'xgboost__colsample_bytree': [0.6, 0.8, 1.0],
            'xgboost__max_depth': [3, 4, 5]
        }
        return pipeline, param_grid


    pipeline = Pipeline(steps=[(model, classifier)])
    param_grid = {f'{model}__{key}': value for key, value in params.items()}
    return pipeline, param_grid

def training_and_testing(index, antibiotic_name, random_state=42):
    label = each_labels[index]
    X_train, X_test, Y_train, Y_test = train_test_split(X, label, test_size=RATIO_OF_TEST)

    model_names = ['lightgbm', 'catboost', 'xgboost']
    avg_cv_scores = []

    for model_name in model_names:
        pipeline, param_grid = get_pipeline_and_parameters(model_name, random_state)
        grid_search = GridSearchCV(pipeline, param_grid, cv=5, scoring='accuracy')
        grid_search.fit(X_train, Y_train)
        best_model = grid_search.best_estimator_
        predictions = best_model.predict(X_test)
        accuracy = accuracy_score(Y_test, predictions)
    
        cv_scores = cross_val_score(pipeline, X_train, Y_train, cv=5, scoring='accuracy')
        mean_cv_score = np.mean(cv_scores)
        avg_cv_scores.append(mean_cv_score)
        
    print(f'{antibiotic_name} 平均交叉驗證準確率:')
    for model_name, avg_score in zip(model_names, avg_cv_scores):
        print(f'{model_name}: {avg_score:.2f}')


if __name__ == '__main__':
    preprocessing_X()
    every_antibiotic()

