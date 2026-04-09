import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import r2_score

# ---------------- LOAD DATA ----------------
data = pd.read_csv("student.csv")

# ---------------- FEATURES & TARGET ----------------
X = data[['hours_study', 'attendance', 'previous_score']]
y = data['final_score']

# ---------------- TRAIN TEST SPLIT ----------------
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# ---------------- MODELS ----------------
lr = LinearRegression()
dt = DecisionTreeRegressor(random_state=42)
rf = RandomForestRegressor(random_state=42)

# ---------------- TRAINING ----------------
lr.fit(X_train, y_train)
dt.fit(X_train, y_train)
rf.fit(X_train, y_train)

# ---------------- USER INPUT ----------------
print("\nEnter student details:")

hours = float(input("Hours of Study: "))
attendance = float(input("Attendance (%): "))
previous = float(input("Previous Score: "))

# ---------------- INPUT VALIDATION ----------------
if not (0 <= hours <= 24):
    print("❌ Invalid Hours! Enter between 0 and 24")
elif not (0 <= attendance <= 100):
    print("❌ Invalid Attendance! Enter between 0 and 100")
elif not (0 <= previous <= 100):
    print("❌ Invalid Previous Score! Enter between 0 and 100")
else:
    input_data = pd.DataFrame(
        [[hours, attendance, previous]],
        columns=['hours_study', 'attendance', 'previous_score']
    )

    # ---------------- PREDICTION ----------------
    lr_pred = lr.predict(input_data)[0]
    dt_pred = dt.predict(input_data)[0]
    rf_pred = rf.predict(input_data)[0]

    print("\nPredicted Final Scores:")
    print("Linear Regression:", round(lr_pred, 2))
    print("Decision Tree:", round(dt_pred, 2))
    print("Random Forest:", round(rf_pred, 2))

    # ---------------- TEST PREDICTIONS FOR GRAPH ----------------
    lr_pred_test = lr.predict(X_test)
    dt_pred_test = dt.predict(X_test)
    rf_pred_test = rf.predict(X_test)

    # ---------------- ACCURACY ----------------
    lr_acc = r2_score(y_test, lr_pred_test)
    dt_acc = r2_score(y_test, dt_pred_test)
    rf_acc = r2_score(y_test, rf_pred_test)

    print("\nModel Accuracy (R2 Score):")
    print("Linear Regression:", round(lr_acc, 2))
    print("Decision Tree:", round(dt_acc, 2))
    print("Random Forest:", round(rf_acc, 2))

    # ---------------- GRAPH 1 ----------------
    plt.figure()
    plt.scatter(y_test, lr_pred_test)
    plt.xlabel("Actual Scores")
    plt.ylabel("Predicted Scores")
    plt.title("Actual vs Predicted (Linear Regression)")
    plt.show()

    # ---------------- GRAPH 2 ----------------
    models = ['Linear Regression', 'Decision Tree', 'Random Forest']
    scores = [lr_acc, dt_acc, rf_acc]

    plt.figure()
    plt.bar(models, scores)
    plt.xlabel("Models")
    plt.ylabel("R2 Score")
    plt.title("Model Accuracy Comparison")
    plt.show()

    # ---------------- GRAPH 3: FEATURE VS FINAL SCORE ----------------

    plt.figure()
    plt.scatter(data['previous_score'], data['final_score'])
    plt.xlabel("Previous Score")
    plt.ylabel("Final Score")
    plt.title("Previous Score vs Final Score")
    plt.show()