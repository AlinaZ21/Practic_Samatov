import streamlit as st
import pandas as pd
import numpy as np
from sklearn import svm
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, confusion_matrix, roc_auc_score, ConfusionMatrixDisplay
from sklearn.utils import resample
from sklearn import svm
import matplotlib.pyplot as plt


data = pd.read_csv('../data/loan_approval_dataset.csv')
st.write("Датасет 'Одобрение Кредита':", data)

button_clicked_info = st.button("Информация о датасете")

if button_clicked_info:
    st.header("О датасете")
    multi = '''
    Набор финансовых записей и связанной с ними информации, используемой 
    для определения права отдельных лиц или организаций на получение кредитов 
    от кредитного учреждения. Он включает в себя различные факторы, такие как 
    гражданский рейтинг, доход, статус занятости, срок кредита, сумма кредита, 
    стоимость активов и статус кредита. 

    Столбцы:

    * количество иждивенцев
    * образование 
    * статус занятости
    * годовой доход
    * величина займа
    * срок кредита в годах
    * кредитный рейтинг
    * имущественные, коммерческие и другие активы
    '''
    st.markdown(multi)

# delete column id
data = data.drop(['loan_id'], axis=1)
# rename column
renamed_columns = {}
for column in data.columns:
    renamed_columns[column] = column.strip()
data = data.rename(columns = renamed_columns)
# from object to int
data['education'] = data['education'].replace(" Graduate", 1)
data['education'] = data['education'].replace(" Not Graduate", 0)
data['self_employed'] = data['self_employed'].replace(" Yes", 1)
data['self_employed'] = data['self_employed'].replace(" No", 0)
data['loan_status'] = data['loan_status'].replace(" Approved",1)
data['loan_status'] = data['loan_status'].replace(" Rejected",0)

f = lambda x : x/1000000
col_million = ['income_annum', 'loan_amount', 'residential_assets_value', 'commercial_assets_value', 'luxury_assets_value', 'bank_asset_value']
for column in col_million: 
    data[column] = data[column].transform(f)
# up sampling
df_0 = resample(data.loc[data['loan_status']==0],
             replace=True,
             n_samples=len(data.loc[data['loan_status']==1]),
             random_state=42)
df = pd.concat([data.loc[data['loan_status']==1], df_0])
# обучающая и тестовая выборки
y = pd.DataFrame(df['loan_status'])
x = pd.DataFrame(df.drop(['loan_status'], axis=1))
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.3)

st.title("Support Vector Machines polinom")
st.write("Предобработанные данные датасета были разделены на обучающую и тестовую выборки.")
st.write("Обучаем классификатор SVM с полиномиальным ядром на данном датасете.")
# классификатор
svc = svm.SVC(kernel='poly')
svc.fit(x_train, y_train)
y_pred_svc = svc.predict(x_test)
# метрики
st.write("## Оценка качества классификатора")
st.write(f"Точность: {accuracy_score(y_test, y_pred_svc)}")
st.write(f"Площадь под ROC-кривой (AUC): {roc_auc_score(y_test['loan_status'], y_pred_svc)}")


cm = confusion_matrix(y_test['loan_status'],y_pred_svc)
disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=svc.classes_)
disp.plot()
plt.show()
st.write("Матрица неточностей:")
st.pyplot(plt)
