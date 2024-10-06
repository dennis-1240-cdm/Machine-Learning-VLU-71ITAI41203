import streamlit as st
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.naive_bayes import BernoulliNB, MultinomialNB
from sklearn.metrics import accuracy_score

# Tiêu đề ứng dụng
st.title('Naive Bayes Classification - Bernoulli vs Multinomial')

# Upload file CSV
uploaded_file = st.file_uploader("Upload CSV file", type=["csv"])

if uploaded_file is not None:
    # Đọc dữ liệu từ file CSV
    data = pd.read_csv(uploaded_file)
    
    # Hiển thị dữ liệu đầu vào
    st.subheader('Dữ liệu đầu vào:')
    st.write(data.head())

    # Xử lý dữ liệu dưới dạng Text
    X = data['Text']
    # Chuyển nhãn về dạng số
    y = data['Label'].map({'positive': 1, 'negative': 0})
    
    st.subheader('Dữ liệu sau khi chuyển đổi nhãn:')
    st.write(y.head())

    # Chia tập dữ liệu để train và test
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # Sử dụng CountVectorizer để chuyển đổi văn bản thành ma trận đặc trưng
    vectorizer = CountVectorizer(binary=False)
    X_train_vec = vectorizer.fit_transform(X_train)
    X_test_vec = vectorizer.transform(X_test)

    # Áp dụng Bernoulli Naive Bayes
    bernoulli_nb = BernoulliNB()
    bernoulli_nb.fit(X_train_vec, y_train)
    y_pred_bernoulli = bernoulli_nb.predict(X_test_vec)

    # Áp dụng Multinomial Naive Bayes
    multinomial_nb = MultinomialNB()
    multinomial_nb.fit(X_train_vec, y_train)
    y_pred_multinomial = multinomial_nb.predict(X_test_vec)

    # Độ chính xác của 2 giải thuật
    accuracy_bernoulli = accuracy_score(y_test, y_pred_bernoulli)
    accuracy_multinomial = accuracy_score(y_test, y_pred_multinomial)

    # Hiển thị kết quả
    st.subheader('Kết quả của 2 giải thuật:')
    st.write(f"Độ chính xác Bernoulli Naive Bayes: {accuracy_bernoulli}")
    st.write(f"Độ chính xác Multinomial Naive Bayes: {accuracy_multinomial}")
else:
    st.write("Vui lòng upload file CSV để tiếp tục.")