import streamlit as st 
import pandas as pd
from sklearn import datasets
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, confusion_matrix
import seaborn as sns
import matplotlib.pyplot as plt


# title
st.set_page_config(page_title="Iris-Classification", 
                   initial_sidebar_state="expanded",
                   page_icon=":sunflower",
                   layout="centered",
                   )

# main body
st.header("Isir Classification")
st.write("This app predcts the Iris flower type")

#side bar
st.sidebar.header("User Input Parameters")

# function for data

# st.slider(label, min_value=None, max_value=None, value=None)

def user_input_features():

    sepal_length = st.sidebar.slider('Sepal length', 4.3, 7.9, 5.4)
    sepal_width = st.sidebar.slider('Sepal width', 2.0, 4.4, 3.4)
    petal_length = st.sidebar.slider('Petal length', 1.0, 6.9, 3.3)
    petal_width = st.sidebar.slider('Petal width', 0.1, 2.5, 0.9)
    
    data = {'sepal_length': sepal_length,
            'sepal_width': sepal_width,
            'petal_length': petal_length,
            'petal_width': petal_width}
    features = pd.DataFrame(data, index=[0])
    return features

test_data = user_input_features()
st.subheader('User Input parameters')
st.dataframe(test_data, hide_index=True)

# machine learning part
iris = datasets.load_iris()
X = iris.data
y = iris.target

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

clf = RandomForestClassifier(n_estimators=100)
clf.fit(X_train, y_train)

y_pred = clf.predict(X_test)

prediction = clf.predict(test_data)
prediction_proba = clf.predict_proba(test_data)

# display the results
st.subheader('Class labels and their corresponding index number')
st.dataframe(
    pd.DataFrame({
        'class labels': iris.target_names
    })
)

st.subheader('Prediction')
st.write(iris.target_names[prediction])
# st.write(prediction)

st.subheader('Prediction Probability')
st.write(prediction_proba)


# # Performace measurements for the classification model
# cm = confusion_matrix(y_test, y_pred)

# # Compute the accuracy score
# accuracy = accuracy_score(y_test, y_pred)

# # Plot the confusion matrix
# fig, ax = plt.subplots(figsize=(3, 2))
# sns.heatmap(cm, annot=True, cmap='Blues', fmt='g', cbar=False, ax=ax)

# # Set labels and title
# ax.set_xlabel('Predicted labels')
# ax.set_ylabel('True labels')
# ax.set_title('Confusion Matrix')

# # Display the plot in Streamlit
# st.pyplot(fig)

# # Display accuracy score
# st.write("Accuracy Score:", accuracy)

