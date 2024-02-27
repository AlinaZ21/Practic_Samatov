import streamlit as st
import pandas as pd
import numpy as np
from sklearn import svm


data = pd.read_csv('../data/possum.csv')
st.write("Датасет 'Одобрение Кредита':", data)

button_clicked_info = st.button("Информация о датасете")

if button_clicked_info:
    st.header("О датасете")
    multi = '''
    Пакет данных опоссумов состоит из девяти морфометрических измерений 
    каждого из 104 горных опоссумов, пойманных в ловушку в семи местах 
    от Южной Виктории до центрального Квинсленда.
    *Вдохновение: Можете ли вы использовать свои навыки регрессии, 
    чтобы предсказать возраст опоссума, длину его головы, будь то самец или самка?* 

    Столбцы:

    * место гдебыл пойман опоссум
    * место обитания (Виктория или другое)
    * пол
    * возраст
    * длина головы
    * ширина черепа
    * общая длина
    * длина хвоста
    * длина ноги
    '''
    st.markdown(multi)

# delete column id
data = data.drop(['case'], axis=1)
# rename column
data = data.rename(columns = {"Pop": "pop"})
data = data.rename(columns = {"sex": "sex_m"})
# from object to int
data['pop'] = data['pop'].replace("Vic", 1)
data['pop'] = data['pop'].replace("other", 0)
data['sex_m'] = data['sex_m'].replace("m", 1)
data['sex_m'] = data['sex_m'].replace("f", 0)
def f(df):
    if df['sex_m'] == 1:
        return 0
    else: return 1
data['sex_f'] = data.apply(f, axis=1)



# обучающая и тестовая выборки
y = pd.DataFrame(df['loan_status'])
x = pd.DataFrame(df.drop(['loan_status'], axis=1))
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.3)

st.title("Support Vector Machines polinom")
st.write("Предобработанные данные датасета были разделены на обучающую и тестовую выборки.")
st.write("Обучаем классификатор SVM с полиномиальным ядром на данном датасете.")
# классификатор
svc = svm.SVR(kernel='poly')
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