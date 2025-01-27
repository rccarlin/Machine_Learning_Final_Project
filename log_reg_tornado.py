import random

import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn import metrics
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split

pd.set_option("display.max_columns", None)
random.seed(2023)

raw = pd.read_csv("tornadoes_FINAL.csv")

# removing columns with way too many missing/ outdated to be addressed, as well as date columns
raw.drop(["PK_WIND_DIR", "PK_WIND_KT", "SL_PRESSURE", "MONTH", "DAY", "TIME", "STATION"], axis=1, inplace= True)

rawY = raw[["YEAR", "TORNADO"]]
rawX = raw.drop("TORNADO", axis=1)


# make dummies

# character: Gusts vs squalls
char_dum = pd.get_dummies(raw.WIND_CHAR)
# condition: CLEAR, FEW, ETC
con_desc_1_dum = pd.get_dummies(rawX.SKY_CON_1_DESC).rename(columns= {"BKN": "BKN1", "CLR": "CLR1", "FEW": "FEW1",
                                                                      "OVC": "OVC1", "SCT": "SCT1"})
con_desc_2_dum = pd.get_dummies(rawX.SKY_CON_2_DESC).rename(columns= {"BKN": "BKN2", "FEW": "FEW2",
                                                                      "OVC": "OVC2", "SCT": "SCT2"})
con_desc_3_dum = pd.get_dummies(rawX.SKY_CON_3_DESC).rename(columns= {"BKN": "BKN3", "FEW": "FEW3",
                                                                      "OVC": "OVC3", "SCT": "SCT3"})
# cloud type
cloud_dum = pd.get_dummies(rawX.CLOUD_TYPE)

# putting the dummies in and taking their original columns out
tornadoes = pd.concat([rawX, char_dum, con_desc_1_dum, con_desc_2_dum, con_desc_3_dum, cloud_dum], axis=1)
tornadoes.drop(["WIND_CHAR", "SKY_CON_1_DESC", "SKY_CON_2_DESC", "SKY_CON_3_DESC", "CLOUD_TYPE"], axis=1, inplace=True)

years = [2011, 2012, 2017, 2019, 2022]

# A loop that allows the user to run multiple experiments per running of the code
# Options include ways to address the missing data and whether or not to make the training/ test set 50/50
# tornadoes and non-tornadoes
user_input = "1"
while input != "7":
    print("Options:")
    print("1: Ignore missing; no limit on non-tornado examples")
    print("2: Ignore missing; limit on non-tornado examples")
    print()
    print("3: Replace missing with medians; no limit on non-tornado examples")
    print("4: Replace missing with medians; limit on non-tornado examples")
    print()
    print("5: Replace missing with various methods; no limit on non-tornado examples")
    print("6: Replace missing with various methods; limit on non-tornado examples")
    print("7: Quit")

    user_input = input("Choice: ")

    if user_input == "7":
        print("goodbye ^-^")
        break

    if user_input == "1" or user_input == "2":  # ignore columns with missing values
        torn = tornadoes.drop(["SKY_CON_1_FT", "SKY_CON_2_FT", "SKY_CON_3_FT", "ALTIMETER"], axis=1)
    elif user_input == "3" or user_input == "4":  # replace missing values with medians
        torn = tornadoes.copy()
        torn["SKY_CON_1_FT"] = torn["SKY_CON_1_FT"].fillna(torn["SKY_CON_1_FT"].median())
        torn["SKY_CON_2_FT"] = torn["SKY_CON_2_FT"].fillna(torn["SKY_CON_2_FT"].median())
        torn["SKY_CON_3_FT"] = torn["SKY_CON_3_FT"].fillna(torn["SKY_CON_3_FT"].median())
        torn["ALTIMETER"] = torn["ALTIMETER"].fillna(torn["ALTIMETER"].median())
    # My way: median makes sense for pressure (since it doesn't seem to change too dramatically
    # but filling in a cloud layer is saying that there are clouds that weren't reported, and that seems sketch
    elif user_input == "5" or user_input == "6":
        torn = tornadoes.copy()
        torn["SKY_CON_1_FT"] = torn["SKY_CON_1_FT"].fillna(300)  # pretends the clouds are very high up
        torn["SKY_CON_2_FT"] = torn["SKY_CON_2_FT"].fillna(300)
        torn["SKY_CON_3_FT"] = torn["SKY_CON_3_FT"].fillna(300)
        torn["ALTIMETER"] = torn["ALTIMETER"].fillna(torn["ALTIMETER"].median())

    # These store the accuracies and recalls for each test year, for comparison later
    year_acc = list()
    year_rec = list()

    for year in years:  # each year gets its chance to be the test set
        print(year, "is now the test set.")

        X = torn[torn.YEAR != year].copy()
        y = rawY[rawY.YEAR != year]["TORNADO"]

        if user_input == "2" or user_input == "4" or user_input == "6":  # need to limit non examples
            torn_idx = y[y == 1].index.tolist()  # where are the tornadoes in the training set?
            non_torn_idx = y[y != 1].index.tolist()  # where are the non-tornadoes?

            num_torn = len(torn_idx)
            non_torn_sample = random.sample(non_torn_idx, k=num_torn) # randomly picks non-tornado examples
            total_sample = torn_idx + non_torn_sample  # gives us a 50/50 positive/negative split
            total_sample.sort()  # mixes the tornadoes back into the non-tornadoes

            X = torn.iloc[total_sample].copy()
            y = rawY.iloc[total_sample]["TORNADO"]

        X.drop("YEAR", axis=1, inplace=True)

        # picking input variables by finding the most correlated to the target variable
        X_train, X_dev, y_train, y_dev = train_test_split(X, y, test_size=.2, random_state=2023, stratify=y)
        temp = pd.concat([y_train, X_train], axis=1)
        correlations = temp.corr(numeric_only=True)["TORNADO"].sort_values().head().index.tolist()

        acc = list()
        rec = list()
        for i in range(1,6):  # experimenting with how many variables to include
            # This loop uses the ith most correlated variables as model inputs and tests on the dev set
            # it will store the accuracy and recall scores so that we can later use the iteration with highest recall

            model = LogisticRegression()
            inn = X_train[correlations[:i]]
            in_dev = X_dev[correlations[:i]]

            model.fit(inn, y_train)

            y_pred = model.predict(in_dev)
            acc.append(metrics.accuracy_score(y_dev, y_pred))
            rec.append(metrics.recall_score(y_dev, y_pred))
            # metrics.ConfusionMatrixDisplay.from_predictions(y_dev, y_pred)
            # plt.show()

        # plotting the dev accuracy and recall over different numbers of variables
        x_ax = [1, 2, 3, 4, 5]
        plt.plot(x_ax, acc, label="Accuracy")
        plt.plot(x_ax, rec, label="Recall")
        plt.legend()
        plt.xticks(x_ax)
        # plt.title("Dev Accuracy and Recall Based on Number of Variables")
        plt.xlabel("Number of Variables Included in the Logistic Regression")
        plt.show() # fixme uncomment

        # working with the number of variables that resulted in the highest recall score
        num_var = np.argsort(rec)[-1] + 1  # +1 because the 0th index is actually 1 variable
        print("Variable choices:", correlations[:num_var])
        model = LogisticRegression()
        X_train = X[correlations[:num_var]]
        model.fit(X_train, y)

        # preparing the test set
        X_test = torn[torn.YEAR == year]
        y_test = rawY[rawY.YEAR == year]["TORNADO"]

        if user_input == "2" or user_input == "4" or user_input == "6":  # need to limit non examples
            torn_idx = y_test[y_test == 1].index.tolist()  # where are the tornadoes in the test set?
            non_torn_idx = y_test[y_test != 1].index.tolist()  # where are the non-tornadoes?

            num_torn = len(torn_idx)
            non_torn_sample = random.sample(non_torn_idx, k=num_torn)
            total_sample = torn_idx + non_torn_sample  # gives us a 50/50 positive/negative split
            total_sample.sort()  # mixes the tornadoes back into the non-tornadoes

            X_test = torn.iloc[total_sample]
            y_test = rawY.iloc[total_sample]["TORNADO"]

        X_test = X_test[correlations[:num_var]]

        y_pred = model.predict(X_test)

        # Performance reports
        print(metrics.classification_report(y_test, y_pred))
        metrics.ConfusionMatrixDisplay.from_predictions(y_test, y_pred)
        plt.show()  # fixme uncomment

        # Adding to the year tracker
        year_acc.append(metrics.accuracy_score(y_test, y_pred))
        year_rec.append(metrics.recall_score(y_test, y_pred))

        # print out some false positives and false negatives
        temp = pd.concat([y_test, X_test], axis=1)
        print(temp[y_pred != y_test])  # fixme uncomment

    # plotting the recall and accuracies over the different years
    plt.scatter(years, year_acc, label="Accuracy")
    plt.scatter(years, year_rec, label="Recall")
    plt.legend()
    plt.xticks(years)
    plt.xlabel("Year")
    plt.show()  # fixme uncomment

    print("Accuracies:", year_acc)
    print("Recall:", year_rec)






