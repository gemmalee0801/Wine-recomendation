import streamlit as st
import pandas as pd
import numpy as np
import networkx as nx
import plotly.graph_objects as go
from streamlit_plotly_events import plotly_events
from itertools import combinations
from sklearn.cluster import KMeans
from sklearn.preprocessing import LabelEncoder
import plotly.express as px
from sklearn.feature_extraction.text import TfidfVectorizer
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from scipy import stats
from gensim.models import Word2Vec
import re

	


data = pd.read_excel(r"C:\Users\gemma\combined_file.xlsx")
df = pd.DataFrame(data)     
df['Food Matching'] = df['Food Matching'].apply(lambda x: re.split(';|,', x))



option = st.sidebar.selectbox(
    'Menu',
     ('home','도수', '음식', '상황'))
if option == 'home' :
    st.title('대학생을 위한 와인 추천')
    st.header("도수/음식/상황에 따른 추천을 제공합니다")
    st.write("와인정보")

    # 드롭다운 메뉴에서 선택된 와인에 따라 필터링
    selected_wine = st.selectbox('와인 선택', df['Wine Name'])
    # 선택된 와인에 대한 정보 표시
    food_pairing = df[df['Wine Name'] == selected_wine]['Food Matching'].values[0]
    Alcohol_Content = df[df['Wine Name'] == selected_wine]['Alcohol Content'].values[0]
    Tasting_Note = df[df['Wine Name'] == selected_wine]['Tasting Note'].values[0]
    st.write(f"설명 : {Tasting_Note}")
    st.write(f"도수 : {Alcohol_Content} ")
    Tasting_Note = df[df['Wine Name'] == selected_wine]['Tasting Note'].values[0]
    st.write(f" 잘 어울리는 음식 : {food_pairing}")



if option == '도수':
    st.title('도수')
    # 문자열 도수를 수치형으로 변환
    def parse_alcohol_content(content):
        try:
            # '%'를 제거하고 범위의 평균값을 계산
            content = content.replace('%', '')
            low, high = map(float, content.split('~'))
            return (low + high) / 2
        except:
            return None

    df['Alcohol Content'] = df['Alcohol Content'].apply(parse_alcohol_content)
    # NaN 값 제거
    df = df.dropna(subset=['Alcohol Content'])

    # 도수 범위 설정
    bins = [4,5,6,7,8,9,10,11, 12, 13, 14, 15,16,17,18]  # 도수 범위 구간 설정
    labels = [f'{bins[i]}~{bins[i+1]}' for i in range(len(bins)-1)]
    df['Alcohol Range'] = pd.cut(df['Alcohol Content'], bins=bins, labels=labels, right=False)

    # 도수 범위별로 와인 집계
    range_summary = df.groupby('Alcohol Range').agg({'Wine Name': lambda x: ', '.join(x), 'Alcohol Content': 'count'}).reset_index()

    # Plotly를 사용하여 도수 범위별 막대 차트 생성
    fig = px.bar(range_summary, x='Alcohol Range', y='Alcohol Content', 
                title='와인 도수 범위별 막대 차트', 
                labels={'Alcohol Range': '도수 범위', 'Alcohol Content': '와인 개수'},
                text='Alcohol Content')

    # 클릭 이벤트 처리
    selected_points = plotly_events(fig)

    if selected_points:
        clicked_range = selected_points[0]['x']
        filtered_wines = df[df['Alcohol Range'] == clicked_range]
        st.subheader(f'도수 범위 {clicked_range}의 와인 목록')
        st.write(filtered_wines[['Wine Name', 'Alcohol Content']])
    else:
        st.write('도수 범위를 클릭하여 해당 도수의 와인 목록을 확인하세요.')

if option == '음식':
    st.title('음식')

    #음식입력
    food_input = st.text_input("음식을 입력하세요:")
    if food_input:
        matching_rows = df[df['Food Matching'].apply(lambda x: food_input in x)]
        

        if not matching_rows.empty:
            st.write(f"'{food_input}'와 관련된 와인의 이름과 설명:")
            st.dataframe(matching_rows[['Wine Name', 'Tasting Note']], height=400)
        else:
            st.write(f"'{food_input}'와 관련된 와인이 없습니다.")
    
    #음식목록
    foodlist = sum(df['Food Matching'], [])
    unique_food_results = list(dict.fromkeys(foodlist))

    cleaned_food_results = []
    for item in unique_food_results:
        cleaned_items = [food.strip() for food in item.split(',')]
        cleaned_food_results.extend(cleaned_items)
    unique_cleaned_food_results = sorted(set(cleaned_food_results))

    # 결과 출력
    st.subheader('음식 목록')
    set_items = list(unique_cleaned_food_results)
    set_items = [item for item in set_items if item != "No matching found"]

    # Streamlit의 columns를 사용하여 가로로 출력
    num_items = len(set_items)
    num_rows = (num_items + 5 ) // 6 # 필요한 행의 수 계산

    columns = st.columns(6)

    for i in range(num_rows):
        for j in range(6):
            index = i * 6 + j
            if index < num_items:
                columns[j].write(set_items[index])
    
    #클러스터링
    df = df[~df['Food Matching'].apply(lambda x: "No matching found" in x)]
    df = df.dropna(subset=['Food Matching'])
    sentences = df['Food Matching'].tolist()

    model = Word2Vec(sentences, vector_size=50, window=5, min_count=1, sg=0)

    # 각 음식 항목을 벡터로 변환
    def get_vector(words):
        # 각 단어의 벡터를 평균내어 문서 벡터를 만듭니다.
        vectors = [model.wv[word] for word in words if word in model.wv]
        if vectors:
            return np.mean(vectors, axis=0)
        else:
            return np.zeros(model.vector_size)

    # 각 음식 항목의 벡터를 계산
    df['Vector'] = df['Food Matching'].apply(lambda x: get_vector(x))
    vectors = np.vstack(df['Vector'].values)
    vec_df = pd.DataFrame(vectors)
    
    model = KMeans(n_clusters=11, random_state=312)
    model.fit(vec_df)
    vec_df['k_means_cluster'] = model.predict(vec_df)
    vec_df['Wine Name']=df['Wine Name']
    vec_df['Food Matching']=df['Food Matching']
    cluster_names = {
    0: "치즈 및 유제품",
    1: "스테이크 및 육류",
    2: "해산물",
    4: "파스타 및 이탈리아 음식",
    5: "샐러드 및 야채",
    6: "바비큐 및 그릴",
    7: "디저트 및 간식",
    8: "아시아 요리",
    9: "피자",
    10: "특이 안주"
   }
    vec_df['Cluster Name'] = vec_df['k_means_cluster'].map(cluster_names)

 
    st.title('Wine and Food Matching by Cluster')

    clusters = vec_df['Cluster Name'].unique()
    selected_cluster = st.selectbox('Select a Clsteuster:', clusters)

    filtered_data = vec_df[vec_df['Cluster Name'] == selected_cluster]
    
    st.dataframe(filtered_data[['Wine Name', 'Food Matching']])


if option == '상황':
    import pandas as pd
    from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
    from wordcloud import WordCloud
    import matplotlib.pyplot as plt
    import seaborn as sns
    from collections import Counter
    import nltk
    from nltk.corpus import stopwords
    import string
    from sklearn.metrics.pairwise import cosine_similarity
    import numpy as np
    from googletrans import Translator
    from deep_translator import GoogleTranslator

    # nltk 데이터 다운로드
    nltk.download('stopwords')

    # 데이터 파일 경로
    file_path = r"C:\Users\gemma\바탕 화면\winemag-data-130k-v2.csv"

    # CSV 파일 읽기
    data = pd.read_csv(file_path)

    # VADER SentimentIntensityAnalyzer 초기화
    analyzer = SentimentIntensityAnalyzer()

    # 불용어와 구두점 제거
    stop_words = set(stopwords.words('english'))
    punctuation = set(string.punctuation)

    # Translator 초기화
    translator = GoogleTranslator()

    # 각 와인의 description을 기반으로 감정 점수 계산
    def calculate_sentiment_scores(description):
        word_sentiments = {
            'joy': 0, 
            'happiness': 0, 
            'neutral': 0, 
            'complexity': 0, 
            'sadness': 0
        }
        
        if pd.isna(description):
            return word_sentiments

        words = description.lower().split()
        words = [word.strip(''.join(punctuation)) for word in words if word not in stop_words and word not in punctuation]
        
        for word in words:
            score = analyzer.polarity_scores(word)['compound']
            if score > 0.5:
                word_sentiments['joy'] += 1
            elif 0.1 < score <= 0.5:
                word_sentiments['happiness'] += 1
            elif -0.1 < score <= 0.1:
                word_sentiments['neutral'] += 1
            elif -0.5 < score <= -0.1:
                word_sentiments['complexity'] += 1
            else:
                word_sentiments['sadness'] += 1

        return word_sentiments

    # 각 와인의 감정 점수 계산하여 새로운 컬럼 추가
    data['sentiment_scores'] = data['description'].apply(calculate_sentiment_scores)

    # 입력 문장의 감정 점수 계산 함수
    def get_sentiment_scores(description):
        return calculate_sentiment_scores(description)

    # 입력 문장의 감정 점수와 유사한 와인 추천 함수
    def recommend_wine(description):
        # 한국어 문장을 영어로 번역
        translated_description = translator.translate(description, src='ko', dest='en')
        
        input_scores = get_sentiment_scores(translated_description)
        input_vector = np.array(list(input_scores.values())).reshape(1, -1)
        
        # 데이터프레임의 감정 점수를 벡터로 변환
        wine_vectors = data['sentiment_scores'].apply(lambda x: np.array(list(x.values())))
        wine_vectors = np.vstack(wine_vectors.values)
        
        # 코사인 유사도를 계산하여 입력 문장과 가장 유사한 와인 찾기
        similarities = cosine_similarity(input_vector, wine_vectors)
        data['similarity'] = similarities[0]
        
        # similarity와 points 기준으로 정렬하여 추천
        recommended_wines = data.sort_values(by=['similarity', 'points'], ascending=[False, False]).head(5)
        return recommended_wines[['title', 'variety', 'points', 'similarity']]

    # 테스트 문장
    test_sentence = st.text_input("상황과 감정을 입력하세요:")

    # 와인 추천
    recommended_wines = recommend_wine(test_sentence)
    print("추천 와인:")
    print(recommended_wines)

    # 추천 결과 시각화
    def visualize_recommendations(recommended_wines):

        # 시각화 1: Barplot
        st.subheader('추천 와인의 유사도 시각화')
        fig, ax = plt.subplots(figsize=(10, 6))
        sns.barplot(x='similarity', y='title', data=recommended_wines, palette='viridis', ax=ax)
        ax.set_xlabel('Similarity')
        ax.set_ylabel('Wine Title')
        ax.set_title('Top 5 Recommended Wines Based on Similarity')
        st.pyplot(fig)

        # 시각화 2: 데이터 테이블
        st.subheader('추천 와인 목록')
        fig, ax = plt.subplots(figsize=(12, 6))
        ax.axis('tight')
        ax.axis('off')
        table = ax.table(cellText=recommended_wines.values, colLabels=recommended_wines.columns, cellLoc='center', loc='center')
        table.auto_set_font_size(False)
        table.set_fontsize(12)
        table.scale(1.2, 1.2)
        st.pyplot(fig)


        # 추천 결과 시각화
    visualize_recommendations(recommended_wines)


    topicdata = pd.read_csv(r"C:\Users\gemma\바탕 화면\result_topicdist.xlsx - Sheet1.csv")
    topic = pd.DataFrame(topicdata)
    # 클러스터 선택 드롭다운
    topics = topic["소속토픽"].unique().tolist()
    selected_topic = st.selectbox("Select a topic:", topics)
    # 선택된 클러스터에 속하는 데이터 필터링
    filtered_data = topic[topic['소속토픽'] == selected_topic]

    # 필터링된 데이터 표시
    st.write(f'Cluster {selected_topic} contains the following wines and their matching foods:')
    st.dataframe(filtered_data[['Wine Name']])





    


