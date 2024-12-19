import requests
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

OMDB_API_KEY = "3d55b56a"

def get_movie_data(title, year=None):
    url = f"http://www.omdbapi.com/?apikey={OMDB_API_KEY}&t={title}&type=movie&r=json"
    if year:
        url += f"&y={year}"
    response = requests.get(url)
    if response.status_code == 200:
        data = response.json()
        if data['Response'] == 'True' and data.get('Type', '') == 'movie' and 'Short' not in data.get('Genre', '') and 'Documentary' not in data.get('Genre', '') and 'News' not in data.get('Genre', ''):
            return data
    return None

def get_movies_by_keyword(keyword):
    url = f"http://www.omdbapi.com/?apikey={OMDB_API_KEY}&s={keyword}&type=movie&r=json"
    response = requests.get(url)
    if response.status_code == 200:
        data = response.json()
        if data['Response'] == 'True':
            return [movie for movie in data['Search'] if movie.get('Type', '').lower() not in ['short', 'video']]
    return []

def recommend_similar_movies(target_movie_title, target_movie_year):
    target_movie_data = get_movie_data(target_movie_title, target_movie_year)
    if not target_movie_data:
        print("영화를 찾을 수 없습니다. 다시 시도해 주세요.")
        return

    target_director = target_movie_data.get('Director', 'N/A').split(', ')
    target_actors = target_movie_data.get('Actors', 'N/A').split(', ')[:3]
    target_genre = target_movie_data.get('Genre', 'N/A').split(', ')
    target_plot = target_movie_data.get('Plot', '').lower()
    target_year = int(target_movie_data.get('Year', '0')) if target_movie_data.get('Year', '0').isdigit() else None

    search_keywords = target_director + target_actors + target_genre
    related_movies = []
    for keyword in search_keywords:
        related_movies.extend(get_movies_by_keyword(keyword))

    seen_titles = set()
    unique_movies = []
    for movie in related_movies:
        movie_title = movie['Title']
        movie_year = int(movie.get('Year', '0')) if movie.get('Year', '0').isdigit() else None

        if target_year and movie_year:
            if abs(target_year - movie_year) > 10:
                continue

        if movie_title.lower() in seen_titles or movie_title.lower() == target_movie_title.lower():
            continue
        seen_titles.add(movie_title.lower())
        unique_movies.append(movie)

    tfidf_features = []
    metadata_features = []
    target_metadata = target_director + target_actors + target_genre

    tfidf_features.append(target_plot)
    metadata_features.append(target_metadata)

    for movie in unique_movies:
        movie_data = get_movie_data(movie['Title'])
        if movie_data:
            tfidf_features.append(movie_data.get('Plot', '').lower())
            movie_metadata = movie_data.get('Director', 'N/A').split(', ') + movie_data.get('Actors', 'N/A').split(', ')[:3] + movie_data.get('Genre', 'N/A').split(', ')
            metadata_features.append(movie_metadata)
        else:
            tfidf_features.append('')
            metadata_features.append([])

    tfidf_vectorizer = TfidfVectorizer(stop_words='english')
    tfidf_matrix = tfidf_vectorizer.fit_transform(tfidf_features)
    plot_similarities = cosine_similarity(tfidf_matrix[0:1], tfidf_matrix[1:]).flatten()

    metadata_similarities = []
    for i in range(1, len(metadata_features)):
        common_elements = set(target_metadata) & set(metadata_features[i])
        metadata_similarity = len(common_elements) / len(set(target_metadata) | set(metadata_features[i])) if len(set(target_metadata) | set(metadata_features[i])) > 0 else 0
        metadata_similarities.append(metadata_similarity)

    plot_weight = 0.6
    metadata_weight = 0.4
    final_scores = [(unique_movies[i]['Title'], plot_weight * plot_similarities[i] + metadata_weight * metadata_similarities[i]) for i in range(len(unique_movies))]

    recommendations = sorted(final_scores, key=lambda x: x[1], reverse=True)[:5]

    if not recommendations:
        print("추천할 유사한 영화가 없습니다.")
    else:
        print(f"\n'{target_movie_title}'와(과) 유사한 영화 추천:")
        for rec in recommendations:
            print(f"- {rec[0]}")

def main():
    title = input("영화 제목을 입력하세요: ")
    year = input("영화 개봉 연도를 입력하세요: ")
    recommend_similar_movies(title, year)

if __name__ == "__main__":
    main()
