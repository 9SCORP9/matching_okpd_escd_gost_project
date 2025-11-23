import re
import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from sentence_transformers import SentenceTransformer
import inspect
from collections import namedtuple

if not hasattr(inspect, "getargspec"):
    def _getargspec(func):
        full = inspect.getfullargspec(func)
        ArgSpec = namedtuple("ArgSpec", ["args", "varargs", "keywords", "defaults"])
        return ArgSpec(full.args, full.varargs, full.varkw, full.defaults)
    inspect.getargspec = _getargspec

# Теперь безопасно импортируем pymorphy2
try:
    import pymorphy2
    morph = pymorphy2.MorphAnalyzer()
    PYMORPHY_AVAILABLE = True
except Exception as e:
    print("Не удалось загрузить pymorphy2 (лемматизация отключена). Ошибка:", e)
    morph = None
    PYMORPHY_AVAILABLE = False

# === Параметры ===
INPUT_FILE = "okpd_escd_gost_alone.xlsx"
OKPD_SHEET = "okpd2"
ESCD_SHEET = "escd"
GOST_SHEET = "gost"
OKPD_CUTTED_SHEET = "okpd2_cutted"
ESCD_CUTTED_SHEET = "escd_cutted_2"
CODE_COL = "код"
NAME_COL = "имя"
OUTPUT_FILE = "okpd_escd_match.xlsx"
SIMILARITY_THRESHOLD = 0.8  # Порог схожести для включения в результаты
MAX_MATCHES = 4  # Максимальное количество совпадений
SEM_PARAMETR = 0.85 # множитель оценки

# === Препроцессинг ===
abbrev_map = {
    'т.п.': 'технологическое производство',
    'т/п': 'технологическое производство',
    'эл.': 'электрический',
    'проч.': 'прочий',
    'в т.ч.': 'в том числе',
}

# Словарь для нормализации ключевых терминов
term_normalization = {
    'суда': 'судно',
    'корабли': 'корабль',
    'лодки': 'лодка',
    'катера': 'катер',
    'баржи': 'баржа',
    'паромы': 'паром',
    'танкеры': 'танкер',
    'буксиры': 'буксир',
    'плоты': 'плот',
    'понтоны': 'понтон',
    'шлюпки': 'шлюпка',
    'прогулочные': 'прогулочный',
    'спортивные': 'спортивный',
    'морские': 'морской',
    'речные': 'речной',
    'самоходные': 'самоходный',
    'несамоходные': 'несамоходный',
    'пассажирские': 'пассажирский',
    'грузовые': 'грузовой',
}

def normalize_text(s):
    s = str(s).lower()

    # Замена дефисов и обработка пунктуации
    s = s.replace('-', ' ')
    s = re.sub(r'[^\w\s\d]', ' ', s)

    # Развертывание аббревиатур
    for a, full in abbrev_map.items():
        s = s.replace(a, full)

    # Нормализация пробелов
    s = re.sub(r'\s+', ' ', s).strip()

    # Лемматизация (если доступна)
    if PYMORPHY_AVAILABLE and morph is not None:
        tokens = s.split()
        lemmas = []
        for t in tokens:
            # Применяем нормализацию терминов перед лемматизацией
            normalized_token = term_normalization.get(t, t)
            try:
                p = morph.parse(normalized_token)[0]
                lemmas.append(p.normal_form)
            except Exception:
                lemmas.append(normalized_token)
        return ' '.join(lemmas)
    else:
        # Без лемматизации, но с нормализацией терминов
        tokens = s.split()
        normalized_tokens = [term_normalization.get(t, t) for t in tokens]
        return ' '.join(normalized_tokens)

# === Функция для вычисления точного совпадения ключевых слов ===
def keyword_match_score(okpd_text, escd_text):
    """Вычисляет оценку совпадения по ключевым словам"""
    okpd_words = set(okpd_text.split())
    escd_words = set(escd_text.split())

    if not okpd_words or not escd_words:
        return 0

    # Ищем совпадения ключевых терминов
    key_terms = ['судно', 'корабль', 'лодка', 'катер', 'баржа', 'паром',
                'танкер', 'буксир', 'плот', 'понтон', 'шлюпка', 'прогулочный',
                'спортивный', 'морской', 'речной', 'самоходный', 'несамоходный']

    okpd_key_terms = okpd_words.intersection(key_terms)
    escd_key_terms = escd_words.intersection(key_terms)

    # Если нет совпадающих ключевых терминов, снижаем оценку
    key_term_bonus = 1.0 if okpd_key_terms.intersection(escd_key_terms) else 0.5

    # Основное совпадение слов
    common_words = okpd_words.intersection(escd_words)
    union_words = okpd_words.union(escd_words)

    if not union_words:
        return 0

    jaccard_similarity = len(common_words) / len(union_words)

    return jaccard_similarity * key_term_bonus

# === Загрузка данных ===
#df_ok = pd.read_excel(INPUT_FILE, sheet_name=OKPD_SHEET)
#df_es = pd.read_excel(INPUT_FILE, sheet_name=ESCD_SHEET)

df_ok = pd.read_excel(INPUT_FILE, sheet_name=OKPD_CUTTED_SHEET)
#df_es = pd.read_excel(INPUT_FILE, sheet_name=ESCD_CUTTED_SHEET)
df_es = pd.read_excel(INPUT_FILE, sheet_name=GOST_SHEET)

# Удаляем пустые строки
df_ok = df_ok.dropna(subset=[NAME_COL])
df_es = df_es.dropna(subset=[NAME_COL])

df_ok['norm'] = df_ok[NAME_COL].astype(str).apply(normalize_text)
df_es['norm'] = df_es[NAME_COL].astype(str).apply(normalize_text)

print(f"OKPD rows: {len(df_ok)}, ESCD rows: {len(df_es)}")
print("Примеры нормализации:")
for i in range(min(3, len(df_ok))):
    print("OKPD:", df_ok.loc[i, NAME_COL], "->", df_ok.loc[i, 'norm'])

# === TF-IDF ===
print("Вычисление TF-IDF векторов...")
tfidf = TfidfVectorizer(ngram_range=(1, 2), min_df=1, max_df=0.8)
all_texts = pd.concat([df_ok['norm'], df_es['norm']])
tfidf.fit(all_texts)
tfidf_ok = tfidf.transform(df_ok['norm'])
tfidf_es = tfidf.transform(df_es['norm'])

# === Bi-encoder (sentence-transformers) ===
print("Загрузка bi-encoder модели...")
model = SentenceTransformer("sentence-transformers/paraphrase-multilingual-mpnet-base-v2")
print("Кодирование OKPD текстов...")
emb_ok = model.encode(df_ok['norm'].tolist(), show_progress_bar=True)
print("Кодирование ESCD текстов...")
emb_es = model.encode(df_es['norm'].tolist(), show_progress_bar=True)

# === Улучшенный алгоритм сравнения ===
print("Сравнение OKPD и ESCD записей...")
results = []

for i, row in df_ok.iterrows():
    okpd_code = row[CODE_COL]
    okpd_name = row[NAME_COL]
    okpd_norm = row['norm']

    # Вычисляем семантическую схожесть
    vec_ok = emb_ok[i].reshape(1, -1)
    sem_scores = cosine_similarity(vec_ok, emb_es).flatten()

    # Вычисляем TF-IDF схожесть
    tfidf_scores = cosine_similarity(tfidf_ok[i], tfidf_es).flatten()

    # Комбинированная оценка
    combined_scores = SEM_PARAMETR * sem_scores + (1-SEM_PARAMETR) * tfidf_scores

    # Добавляем бонус за совпадение ключевых слов
    for j in range(len(combined_scores)):
        keyword_score = keyword_match_score(okpd_norm, df_es.loc[j, 'norm'])
        combined_scores[j] *= (1 + 0.2 * keyword_score)  # Бонус до 20%

    # Собираем кандидатов выше порога
    candidates = []
    for j, score in enumerate(combined_scores):
        if score >= SIMILARITY_THRESHOLD:
            candidates.append({
                'index': j,
                'score': score,
                'code': df_es.loc[j, CODE_COL],
                'name': df_es.loc[j, NAME_COL]
            })

    # Сортируем по убыванию схожести и берем до MAX_MATCHES
    candidates.sort(key=lambda x: x['score'], reverse=True)
    final_matches = candidates[:MAX_MATCHES]

    # Формируем результат
    out = {
        "OKPD2 Code": okpd_code,
        "OKPD2 Name": okpd_name
    }

    # Добавляем совпадения (0-3)
    for k, match in enumerate(final_matches, 1):
        out[f"ESCD Code {k}"] = match['code']
        out[f"ESCD Name {k}"] = match['name']
        out[f"Similarity Score {k}"] = round(match['score'], 4)

    # Заполняем пустые места, если совпадений меньше 3
    for k in range(len(final_matches) + 1, MAX_MATCHES + 1):
        out[f"ESCD Code {k}"] = ""
        out[f"ESCD Name {k}"] = ""
        out[f"Similarity Score {k}"] = ""

    results.append(out)

    if i % 50 == 0:  # Прогресс каждые 50 строк
        print(f"Обработано {i+1}/{len(df_ok)} строк")

# Создаем DataFrame и сохраняем
res_df = pd.DataFrame(results)

# Переупорядочиваем колонки для лучшей читаемости
column_order = ["OKPD2 Code", "OKPD2 Name"]
for k in range(1, MAX_MATCHES + 1):
    column_order.extend([f"ESCD Code {k}", f"ESCD Name {k}", f"Similarity Score {k}"])

res_df = res_df[column_order]
res_df.to_excel(OUTPUT_FILE, index=False)

print(f"\nСохранено в: {OUTPUT_FILE}")
print(f"Всего обработано: {len(results)} строк OKPD")
print(f"Порог схожести: {SIMILARITY_THRESHOLD}")
print(f"Максимальное количество совпадений: {MAX_MATCHES}")

# Статистика
matches_count = [sum(1 for k in range(1, MAX_MATCHES+1) if row[f"ESCD Code {k}"]) for row in results]
print(f"\nСтатистика совпадений:")
print(f"- Без совпадений: {matches_count.count(0)}")
print(f"- 1 совпадение: {matches_count.count(1)}")
print(f"- 2 совпадения: {matches_count.count(2)}")
print(f"- 3 совпадения: {matches_count.count(3)}")
print(f"- 4 совпадения: {matches_count.count(4)}")