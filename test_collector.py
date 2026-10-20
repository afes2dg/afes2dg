import unittest
import os
import shutil
import tempfile
import json
import email
from email.header import decode_header, make_header
import game_news_collector

class TestGameNewsCollector(unittest.TestCase):

    def setUp(self):
        self.test_dir = tempfile.mkdtemp()
        self.original_target_dir = game_news_collector.TARGET_DIR
        self.original_seen_file = game_news_collector.SEEN_FILE
        game_news_collector.TARGET_DIR = self.test_dir
        game_news_collector.SEEN_FILE = os.path.join(self.test_dir, "seen_news.json")

    def tearDown(self):
        game_news_collector.TARGET_DIR = self.original_target_dir
        game_news_collector.SEEN_FILE = self.original_seen_file
        shutil.rmtree(self.test_dir)

    def test_load_and_save_seen_news(self):
        seen = {"News 1", "News 2"}
        game_news_collector.save_seen_news(seen)
        loaded_seen = game_news_collector.load_seen_news()
        self.assertEqual(seen, loaded_seen)

    def test_collect_unique_news_deduplication(self):
        # Initial run
        run1 = game_news_collector.collect_unique_news(count=5)
        self.assertEqual(len(run1), 5)

        # Second run should collect different news
        run2 = game_news_collector.collect_unique_news(count=5)
        self.assertEqual(len(run2), 5)

        # Check no overlap
        titles_run1 = {item['title'] for item in run1}
        titles_run2 = {item['title'] for item in run2}
        intersection = titles_run1.intersection(titles_run2)
        self.assertEqual(len(intersection), 0, f"Found duplicate titles: {intersection}")

    def test_create_mhtml_file(self):
        html_content = "<html><body><h1>Test Game News</h1></body></html>"
        output_filepath = os.path.join(self.test_dir, "test.mhtml")
        game_news_collector.create_mhtml_file(html_content, output_filepath, "Test Title")

        self.assertTrue(os.path.exists(output_filepath))
        with open(output_filepath, "rb") as f:
            msg = email.message_from_binary_file(f)

        self.assertTrue(msg.is_multipart())
        decoded_subject = str(make_header(decode_header(msg["Subject"])))
        self.assertIn("Test Title", decoded_subject)

if __name__ == "__main__":
    unittest.main()
