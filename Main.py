import pandas as pd
import pickle
from sklearn.neighbors import KNeighborsClassifier
from sklearn import model_selection

data = pd.read_csv("Cleaned-Data.csv")
data = data[["Fever","Tiredness","Dry-Cough","Difficulty-in-Breathing","Sore-Throat","Pains","Nasal-Congestion","Runny-Nose","Diarrhea","Age_0-9","Age_10-19","Age_20-24","Age_25-59","Age_60+","Gender_Female","Gender_Male","Severity_Mild","Severity_Moderate","Severity_None","Severity_Severe","Contact_Dont-Know","Contact_No","Contact_Yes"]]
data = data[0:10000]

y = data["Severity_None"]
x = data.drop("Severity_Mild", axis=1)
x = x.drop("Severity_Moderate", axis=1)
x = x.drop("Severity_None", axis=1)
x = x.drop("Severity_Severe", axis=1)

best = 0
best_history = []
for i in range(10):
    x_train, x_test, y_train, y_test = model_selection.train_test_split(x, y, test_size=0.1)

    kmodel = KNeighborsClassifier(n_neighbors=9)
    kmodel.fit(x_train, y_train)
    acc = kmodel.score(x_test, y_test)

    if acc > best:
        best = acc
        best_history.append(best)
        with open("savedmodel.pickle", "wb") as f:
            pickle.dump(kmodel, f)

    print(i)
print(best)


pickle_in = open("savedmodel.pickle", "rb")
kmodel = pickle.load(pickle_in)
