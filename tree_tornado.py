import random
import pandas as pd
from sklearn import metrics
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split

from sklearn.tree import DecisionTreeClassifier
from sklearn.tree import plot_tree
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.tree import export_graphviz


pd.set_option("display.max_columns", None)
random.seed(2023)

raw = pd.read_csv("tornadoes_FINAL.csv")

# removing unhelpful identification columns
raw.drop(["MONTH", "DAY", "TIME", "STATION"], axis=1, inplace=True)

rawY = raw[["YEAR", "TORNADO"]]
rawX = raw.drop("TORNADO", axis=1)


# make dummies
# character: Gusts vs squalls
char_dum = pd.get_dummies(raw.WIND_CHAR)

# condition: CLEAR, FEW, ETC
con_desc_1_dum = pd.get_dummies(rawX.SKY_CON_1_DESC).rename(columns={"BKN": "BKN1", "CLR": "CLR1", "FEW": "FEW1",
                                                                     "OVC": "OVC1", "SCT": "SCT1"})
con_desc_2_dum = pd.get_dummies(rawX.SKY_CON_2_DESC).rename(columns={"BKN": "BKN2", "FEW": "FEW2",
                                                                     "OVC": "OVC2", "SCT": "SCT2"})
con_desc_3_dum = pd.get_dummies(rawX.SKY_CON_3_DESC).rename(columns={"BKN": "BKN3", "FEW": "FEW3",
                                                                     "OVC": "OVC3", "SCT": "SCT3"})
# cloud type
cloud_dum = pd.get_dummies(rawX.CLOUD_TYPE)

# putting the dummies in and taking their original columns out
tornadoes = pd.concat([rawX, char_dum, con_desc_1_dum, con_desc_2_dum, con_desc_3_dum, cloud_dum], axis=1)
tornadoes.drop(["WIND_CHAR", "SKY_CON_1_DESC", "SKY_CON_2_DESC", "SKY_CON_3_DESC", "CLOUD_TYPE"], axis=1, inplace=True)
years = [2011, 2012, 2017, 2019, 2022]

user_input = "1"
# A loop that allows the user to run multiple experiments per running of the code
# Options include ways to address the missing data, whether or not to make the training/ test set 50/50
# tornadoes and non-tornadoes, and what typeof decision tree to use
while input != "7":
    print("Options:")

    print("1: Basic decision tree; ignore missing; no limit on non-tornado examples")
    print("2: Basic decision tree; ignore missing; limit on non-tornado examples")
    print()
    print("3: Basic decision tree; replace missing with medians; no limit on non-tornado examples")
    print("4: Basic decision tree; replace missing with medians; limit on non-tornado examples")
    print()
    print("5: Basic decision tree; various ways of handling missing; no limit on non-tornado examples")
    print("6: Basic decision tree; various ways of handling missing; limit on non-tornado examples")
    print()
    print("7: Advanced decision tree; don't replace missing; no limit on non-tornado examples")
    print("8: Advanced decision tree; don't replace missing; limit on non-tornado examples")
    print()
    print("9: Advanced decision tree; replace missing with medians; no limit on non-tornado examples")
    print("10: Advanced decision tree; replace missing with medians; limit on non-tornado examples")
    print()
    print("11: Advanced decision tree; various ways of handling missing; no limit on non-tornado examples")
    print("12: Advanced decision tree; various ways of handling missing; limit on non-tornado examples")
    print("13: Quit")


    user_input = input("Choice: ")

    if user_input == "13":  # quit
        print("goodbye ^-^")
        break
    if user_input == "1" or user_input == "2":  # ignore columns with missing values
        torn = tornadoes.drop(["SKY_CON_1_FT", "SKY_CON_2_FT", "SKY_CON_3_FT", "ALTIMETER", "PK_WIND_DIR",
                               "PK_WIND_KT", "SL_PRESSURE"], axis=1)
    elif user_input in ["3", "4", "9", "10"]:  # replace missing values with median
        torn = tornadoes.copy()
        torn["SKY_CON_1_FT"] = torn["SKY_CON_1_FT"].fillna(torn["SKY_CON_1_FT"].median())
        torn["SKY_CON_2_FT"] = torn["SKY_CON_2_FT"].fillna(torn["SKY_CON_2_FT"].median())
        torn["SKY_CON_3_FT"] = torn["SKY_CON_3_FT"].fillna(torn["SKY_CON_3_FT"].median())
        torn["ALTIMETER"] = torn["ALTIMETER"].fillna(torn["ALTIMETER"].median())
        torn["PK_WIND_DIR"] = torn["PK_WIND_DIR"].fillna(torn["PK_WIND_DIR"].median())
        torn["PK_WIND_KT"] = torn["PK_WIND_KT"].fillna(torn["PK_WIND_KT"].median())
        torn["SL_PRESSURE"] = torn["SL_PRESSURE"].fillna(torn["SL_PRESSURE"].median())
    elif user_input in ["5", "6", "11", "12"]:  # addresses missing values as I see fit (see paper)
        torn = tornadoes.copy()
        torn["SKY_CON_1_FT"] = torn["SKY_CON_1_FT"].fillna(300)  # just pretends the clouds are very high up
        torn["SKY_CON_2_FT"] = torn["SKY_CON_2_FT"].fillna(300)
        torn["SKY_CON_3_FT"] = torn["SKY_CON_3_FT"].fillna(300)
        torn["ALTIMETER"] = torn["ALTIMETER"].fillna(torn["ALTIMETER"].median())
        torn.drop(["PK_WIND_DIR", "PK_WIND_KT", "SL_PRESSURE"], axis=1, inplace=True)
    elif user_input == "7" or user_input == "8":
        torn = tornadoes.copy()

    year_acc = list()
    year_rec = list()

    for year in years:  # each year gets its chance to be the test set
        print(year, "is now the test set.")

        X_train = torn[torn.YEAR != year].copy()
        y_train = rawY[rawY.YEAR != year]["TORNADO"]

        if user_input in ["2", "4", "6", "8", "10", "12"]:  # need to limit non examples
            torn_idx = y_train[y_train == 1].index.tolist()  # where are the tornadoes in the training set?
            non_torn_idx = y_train[y_train != 1].index.tolist()  # where are the non-tornadoes?

            num_torn = len(torn_idx)
            non_torn_sample = random.sample(non_torn_idx, k=num_torn)
            total_sample = torn_idx + non_torn_sample  # gives us a 50/50 positive/negative split
            total_sample.sort()  # mixes the tornadoes back into the non-tornadoes

            X_train = torn.iloc[total_sample].copy()
            y_train = rawY.iloc[total_sample]["TORNADO"]
        X_train.drop("YEAR", axis=1, inplace=True)

        # pick tree depth
        acc = list()
        rec = list()
        X_train_a, X_dev, y_train_a, y_dev = train_test_split(X_train, y_train, test_size=.2, random_state=2023, stratify=y_train)
        # Tries a  tree of depth d to see which has the best recall on the dev set
        for d in range(1,31):
            if user_input in ["1", "2", "3", "4", "5", "6"]:  # basic decision tree
                model = DecisionTreeClassifier(max_depth=d)
            else:
                model = HistGradientBoostingClassifier(max_depth=d)
            model.fit(X_train_a, y_train_a)
            y_pred = model.predict(X_dev)
            acc.append(metrics.accuracy_score(y_dev, y_pred))
            rec.append(metrics.recall_score(y_dev, y_pred))

        # plotting accuracy and recall for each depth
        # fixme uncomment
        x_ax = np.arange(1, 31)
        plt.plot(x_ax, acc, label="Accuracy")
        plt.plot(x_ax, rec, label="Recall")
        plt.legend()
        plt.xticks(x_ax)
        # plt.title("Dev Accuracy and Recall with Increasing Tree Depth")
        plt.xlabel("Tree Depth")
        plt.show()

        # Making a model of depth that achieved the best recall
        best_depth = np.argsort(rec)[-1] + 1  # so the -1 will get the biggest, but index 0 corresponds to depth 1
        if user_input in ["1", "2", "3", "4", "5", "6"]:  # basic decision tree
            model = DecisionTreeClassifier(max_depth=best_depth)
        else:
            model = HistGradientBoostingClassifier(max_depth=best_depth)

        model.fit(X_train, y_train)

        # Preparing the test set
        X_test = torn[torn.YEAR == year].copy()
        y_test = rawY[rawY.YEAR == year]["TORNADO"]
        if user_input in ["2", "4", "6", "8", "10", "12"]:  # need to limit non examples for test set
            torn_idx = y_test[y_test == 1].index.tolist()  # where are the tornadoes in the test set?
            non_torn_idx = y_test[y_test != 1].index.tolist()  # where are the non-tornadoes?

            num_torn = len(torn_idx)
            non_torn_sample = random.sample(non_torn_idx, k=num_torn)
            total_sample = torn_idx + non_torn_sample  # gives us a 50/50 positive/negative split
            total_sample.sort()  # mixes the tornadoes back into the non-tornadoes

            X_test = torn.iloc[total_sample].copy()
            y_test = rawY.iloc[total_sample]["TORNADO"]
        X_test.drop("YEAR", axis=1, inplace=True)

        # running the finished model
        y_pred = model.predict(X_test)
        print(metrics.classification_report(y_test, y_pred))
        # fixme uncomment
        metrics.ConfusionMatrixDisplay.from_predictions(y_test, y_pred)
        plt.show()
        year_acc.append(metrics.accuracy_score(y_test, y_pred))
        year_rec.append(metrics.recall_score(y_test, y_pred))

        # if its a basic tree, display tree now
        if user_input in ["1", "2", "3", "4", "5", "6"]:
            # fixme uncomment
            names = list(map(str, model.classes_.tolist()))
            plot_tree(model, feature_names=X_test.columns, filled=True, class_names=names, fontsize=6)
            plt.show()
            temp = pd.Series(model.feature_importances_, index=X_test.columns).sort_values(ascending=False)
            print(temp.head(3))
            print()

        # print out some false positives and false negatives
        temp = pd.concat([y_test, X_test], axis=1)
        print(temp[y_pred != y_test])  # fixme uncomment

    # plotting the recall and accuracies over the different years
    plt.scatter(years, year_acc, label="Accuracy")
    plt.scatter(years, year_rec, label="Recall")
    plt.legend()
    plt.xticks(years)
    plt.xlabel("Year")
    plt.show()

    print("Accuracies:", year_acc)
    print("Recall:", year_rec)






