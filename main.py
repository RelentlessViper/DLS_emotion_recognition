import os
import re
import numpy as np
from pymilvus import connections, FieldSchema, CollectionSchema, DataType, Collection, utility
from PIL import Image
from torchvision import transforms
from Scripts import embedding

def initialize_table():
    fields = [
        FieldSchema(name="id", dtype=DataType.INT64, is_primary=True),
        FieldSchema(name="embedding", dtype=DataType.FLOAT_VECTOR, dim=128),
        FieldSchema(name="class", dtype=DataType.VARCHAR, max_length=20),
        FieldSchema(name="name", dtype=DataType.VARCHAR, max_length=255)
    ]

    schema = CollectionSchema(fields, "Описание коллекции")

    collection = Collection("emotion_collection", schema)
    image_root = "archive/train"
    embedding_root = "train_embeddings"

    data = collect_data(image_root, embedding_root)

    entities = [
        data["id"],
        data["embedding"],
        data["class"],
        data["name"]
    ]

    collection.insert(entities)

    index_params = {
        "index_type": "IVF_FLAT",  # Тип индекса, можно использовать другие типы, такие как IVF_SQ8, HNSW и т.д.
        "metric_type": "L2",  # Тип метрики, можно использовать L2, IP и т.д.
        "params": {"nlist": 128}  # Параметры индекса, зависят от типа индекса
    }
    collection.create_index(field_name="embedding", index_params=index_params)


def read_embeddings(embedding_file):
    with open(embedding_file, 'r') as f:
        embeddings = [list(map(float, line.strip().split())) for line in f]
    return embeddings


def extract_number(filename):
    match = re.search(r'\d+', filename)
    res = int(match.group()) if match else float('inf')
    if filename.startswith('fer'):
        res += 1000000
    return res


def collect_data(image_root, embedding_root):
    data = {
        "id": [],
        "embedding": [],
        "class": [],
        "name": []
    }
    id_counter = 0

    for emotion in os.listdir(image_root):
        image_dir = os.path.join(image_root, emotion)
        embedding_file = os.path.join(embedding_root, emotion, 'embeddings.txt')

        if not os.path.exists(embedding_file):
            continue

        embeddings = read_embeddings(embedding_file)
        image_names = os.listdir(image_dir)
        image_names.sort(key=extract_number)  # Сортировка по числовой части имен файлов
        for img_name, embedding in zip(image_names, embeddings):
            data["id"].append(id_counter)
            data["embedding"].append(embedding)
            data["class"].append(emotion)
            data["name"].append(img_name)
            id_counter += 1

    return data


def search_nearest_neighbors(collection_name, query_embedding, k=5):
    connections.connect("default", host="localhost", port="19530")

    if collection_name not in utility.list_collections():
        print(f"Collection {collection_name} does not exist.")
        return

    collection = Collection(collection_name)
    collection.load()

    search_params = {
        "metric_type": "L2",
        "params": {"nprobe": 10}
    }

    results = collection.search(
        data=[query_embedding],  # Запрос в виде списка
        anns_field="embedding",  # Поле с эмбеддингами
        param=search_params,  # Параметры поиска
        limit=k,  # Количество ближайших соседей
        output_fields=["id", "embedding", "class", "name"]  # Поля для вывода
    )

    for result in results[0]:
        print(
            f"ID: {result.id}, Distance: {result.distance}, Class: {result.entity.get('class')}, Name: {result.entity.get('name')}")

if __name__ == '__main__':
    connections.connect("default", host="localhost", port="19530")
    # dropped = Collection("emotion_collection")
    # dropped.drop()
    # initialize_table()
    collections = utility.list_collections()
    print(f"List all collections:\n", collections)
    collection_name = "emotion_collection"

    collection = Collection(collection_name)
    collection.load()
    results = collection.query(expr="", output_fields=["id", "embedding", "class", "name"], limit=1000)
    embedding_generator = embedding.Embedding_generator("Models/encoder_v1.pth")
    test_embedding = embedding_generator("Data/train/fear/augmented_120.png")
    print(test_embedding)
    # query_embedding = np.random.random(128).astype(np.float32).tolist()
    search_nearest_neighbors("emotion_collection", test_embedding, k=5)
    # Вывод данных
    # for result in results:
    #     print(result)
