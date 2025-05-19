import unittest
import time
import os
from pathlib import Path
from src.index_creater.inverted_index import InvertedIndex, index_initializer
from src.utils.preprocessor import DocumentProcessor

class TestIndexingPerformance(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.test_data_path = Path("data/test_performance")
        cls.test_data_path.mkdir(exist_ok=True)
        
        # Генерация тестового набора данных (40k документов)
        cls.generate_test_docs(40000)

    @staticmethod
    def generate_test_docs(num_docs):
        docs = {}
        for i in range(num_docs):
            text = f"Документ {i} о деятельности ректора СПбГУ" if i % 2 == 0 \
                else f"Документ {i} о научных достижениях МГУ"
            docs[i] = text
        return docs

    def test_compression_impact(self):
        # Конфигурации тестирования
        configs = [
            {'encoding': None, 'name': 'Без сжатия'},
            {'encoding': 'gamma', 'name': 'Gamma-сжатие'},
            {'encoding': 'delta', 'name': 'Delta-сжатие'}
        ]
        
        results = {}
        docs = self.generate_test_docs(40000)
        
        for config in configs:
            processor = DocumentProcessor()
            index = index_initializer(
                database_path=str(self.test_data_path / "test_db.sqlite"),
                preprocessor=processor,
                encoding=config['encoding']
            )
            
            # Замер времени индексирования
            start_time = time.time()
            for doc_id, text in docs.items():
                index.add_document(doc_id, text)
            index.save_index()
            elapsed = time.time() - start_time
            
            # Замер размера индекса
            index_size = os.path.getsize(self.test_data_path / "index" / "index.pkl")
            
            results[config['name']] = {
                'time': elapsed,
                'size': index_size
            }
        
        # Вывод результатов
        print("\nРезультаты тестирования сжатия:")
        for name, data in results.items():
            print(f"{name}:")
            print(f"  Время индексирования: {data['time']:.2f} сек")
            print(f"  Размер индекса: {data['size'] / 1024 / 1024:.2f} МБ")

    def test_search_performance(self):
        docs = self.generate_test_docs(40000)
        processor = DocumentProcessor()
        index = index_initializer(
            database_path=str(self.test_data_path / "test_db.sqlite"),
            preprocessor=processor
        )
        
        # Добавление документов в индекс
        for doc_id, text in docs.items():
            index.add_document(doc_id, text)
        
        # Тестирование скорости поиска
        test_cases = [
            "ректор спбгу",
            "мгу научный достижение",
            "несуществующий запрос"
        ]
        
        results = {}
        for query in test_cases:
            start_time = time.time()
            results = index.search(query)
            elapsed = time.time() - start_time
            print(f"\nПоиск: '{query}'")
            print(f"  Найдено документов: {len(results)}")
            print(f"  Время выполнения: {elapsed:.4f} сек")
            self.assertLess(elapsed, 0.5)


import unittest
from inverted_index import InvertedIndex  # Предполагается, что у вас есть модуль inverted_index с реализацией

class TestInvertedIndex(unittest.TestCase):
    def setUp(self):
        # Создаем тестовые данные
        self.test_docs = {
            "doc1": "Ректор СПбГУ объявил о новых исследованиях",
            "doc2": "В МГУ прошла конференция по искусственному интеллекту",
            "doc3": "СПбГУ и МГУ сотрудничают в области образования"
        }
        self.index = InvertedIndex()

    def test_build_index(self):
        # Тест на создание инвертированного индекса
        self.index.build_index(self.test_docs)
        self.assertIn("Ректор", self.index.index)
        self.assertIn("СПбГУ", self.index.index)
        self.assertIn("МГУ", self.index.index)

    def test_search_query(self):
        # Тест на поиск по запросу
        self.index.build_index(self.test_docs)
        results = self.index.search("Ректор СПбГУ")
        self.assertEqual(results, ["doc1"])

    def test_compression(self):
        # Тест на сжатие индекса
        self.index.build_index(self.test_docs)
        original_size = self.index.get_index_size()
        self.index.apply_compression()
        compressed_size = self.index.get_index_size()
        self.assertLess(compressed_size, original_size)

    def test_compression_efficiency(self):
        # Тест на эффективность сжатия
        self.index.build_index(self.test_docs)
        original_size = self.index.get_index_size()
        self.index.apply_compression()
        compressed_size = self.index.get_index_size()
        efficiency = (original_size - compressed_size) / original_size
        self.assertGreater(efficiency, 0.5)  # Ожидаем, что сжатие уменьшит размер хотя бы на 50%

    def test_search_speed(self):
        # Тест на скорость поиска
        import time
        self.index.build_index(self.test_docs)
        start_time = time.time()
        self.index.search("Ректор СПбГУ")
        search_time = time.time() - start_time
        self.assertLess(search_time, 0.1)  # Ожидаем, что поиск займет менее 0.1 секунды

    def test_large_dataset_indexing(self):
        # Тест на индексирование большого набора данных
        large_docs = {f"doc{i}": "СПбГУ МГУ" for i in range(40000)}
        start_time = time.time()
        self.index.build_index(large_docs)
        indexing_time = time.time() - start_time
        self.assertLess(indexing_time, 60)  # Ожидаем, что индексирование займет менее 60 секунд

    def test_compression_large_dataset(self):
        # Тест на сжатие большого набора данных
        large_docs = {f"doc{i}": "СПбГУ МГУ" for i in range(40000)}
        self.index.build_index(large_docs)
        original_size = self.index.get_index_size()
        self.index.apply_compression()
        compressed_size = self.index.get_index_size()
        self.assertLess(compressed_size, original_size)

    def test_search_large_dataset(self):
        # Тест на поиск в большом наборе данных
        large_docs = {f"doc{i}": "СПбГУ МГУ" for i in range(40000)}
        self.index.build_index(large_docs)
        start_time = time.time()
        self.index.search("Ректор СПбГУ")
        search_time = time.time() - start_time
        self.assertLess(search_time, 0.5)  # Ожидаем, что поиск займет менее 0.5 секунды

    def test_index_consistency(self):
        # Тест на согласованность индекса после сжатия
        self.index.build_index(self.test_docs)
        original_index = self.index.index.copy()
        self.index.apply_compression()
        self.assertEqual(self.index.index, original_index)

  

if __name__ == '__main__':
    unittest.main()
