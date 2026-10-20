#!/usr/bin/env python3
"""
Game News MHTML Collector
Collects 5 new (unseen) game news articles every hour and saves them into an MHTML file
in 'C:\\Users\\Admin\\Desktop\\jules test' with unique timestamp numbering.
"""

import os
import sys
import time
import json
import urllib.request
import urllib.parse
import xml.etree.ElementTree as ET
from datetime import datetime
from email.mime.multipart import MIMEMultipart
from email.mime.text import MIMEText
from email.header import Header

# Target directory specified by user requirement
DEFAULT_DIR = r"C:\Users\Admin\Desktop\jules test"
TARGET_DIR = os.environ.get("TARGET_DIR", DEFAULT_DIR)

SEEN_FILE = os.path.join(TARGET_DIR, "seen_news.json")

# Multiple RSS search queries to ensure a continuous stream of new articles
RSS_QUERIES = [
    "게임",
    "게임 신작",
    "스팀 게임",
    "콘솔 게임",
    "모바일 게임",
    "e스포츠",
    "지스타 게임",
    "게임 개발",
    "온라인 게임",
    "게임 출시",
]

def load_seen_news():
    if os.path.exists(SEEN_FILE):
        try:
            with open(SEEN_FILE, "r", encoding="utf-8") as f:
                return set(json.load(f))
        except Exception as e:
            print(f"[Warning] Failed to load seen news history: {e}")
            return set()
    return set()

def save_seen_news(seen_set):
    try:
        os.makedirs(TARGET_DIR, exist_ok=True)
        with open(SEEN_FILE, "w", encoding="utf-8") as f:
            json.dump(list(seen_set), f, ensure_ascii=False, indent=2)
    except Exception as e:
        print(f"[Error] Failed to save seen news history: {e}")

def fetch_rss_news(query):
    encoded_query = urllib.parse.quote(query)
    url = f"https://news.google.com/rss/search?q={encoded_query}&hl=ko&gl=KR&ceid=KR:ko"
    req = urllib.request.Request(url, headers={"User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64)"})
    items = []
    try:
        with urllib.request.urlopen(req, timeout=10) as resp:
            xml_data = resp.read()
            root = ET.fromstring(xml_data)
            for item in root.findall(".//item"):
                title_elem = item.find("title")
                link_elem = item.find("link")
                pubDate_elem = item.find("pubDate")
                desc_elem = item.find("description")

                title = title_elem.text.strip() if title_elem is not None and title_elem.text else ""
                link = link_elem.text.strip() if link_elem is not None and link_elem.text else ""
                pub_date = pubDate_elem.text.strip() if pubDate_elem is not None and pubDate_elem.text else ""
                description = desc_elem.text.strip() if desc_elem is not None and desc_elem.text else ""

                if title and link:
                    items.append({
                        "title": title,
                        "link": link,
                        "pubDate": pub_date,
                        "description": description
                    })
    except Exception as e:
        print(f"[Error] Fetching RSS for query '{query}': {e}")
    return items

def collect_unique_news(count=5):
    seen_set = load_seen_news()
    collected = []

    for query in RSS_QUERIES:
        if len(collected) >= count:
            break
        items = fetch_rss_news(query)
        for item in items:
            # Use title as key identifier
            news_id = item["title"]
            if news_id not in seen_set:
                seen_set.add(news_id)
                collected.append(item)
                if len(collected) >= count:
                    break

    save_seen_news(seen_set)
    return collected

def generate_html_content(news_items, timestamp_str):
    items_html = ""
    for idx, item in enumerate(news_items, 1):
        items_html += f"""
        <div class="news-card">
            <div class="news-number">{idx}</div>
            <div class="news-content">
                <h2><a href="{item['link']}" target="_blank" rel="noopener noreferrer">{item['title']}</a></h2>
                <div class="meta">발행일: {item['pubDate']}</div>
                <div class="description">{item['description']}</div>
                <div class="link-bar"><a href="{item['link']}" target="_blank">기사 원문 읽기 &rarr;</a></div>
            </div>
        </div>
        """

    html = f"""<!DOCTYPE html>
<html lang="ko">
<head>
    <meta charset="UTF-8">
    <title>게임 뉴스 모음 - {timestamp_str}</title>
    <style>
        body {{
            font-family: 'Malgun Gothic', 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif;
            background-color: #f4f6f9;
            color: #333;
            margin: 0;
            padding: 20px;
        }}
        .container {{
            max-width: 800px;
            margin: 0 auto;
            background: #ffffff;
            padding: 30px;
            border-radius: 10px;
            box-shadow: 0 4px 15px rgba(0, 0, 0, 0.1);
        }}
        header {{
            border-bottom: 2px solid #2c3e50;
            padding-bottom: 15px;
            margin-bottom: 25px;
        }}
        h1 {{
            color: #2c3e50;
            margin: 0 0 10px 0;
            font-size: 24px;
        }}
        .header-meta {{
            color: #7f8c8d;
            font-size: 14px;
        }}
        .news-card {{
            display: flex;
            margin-bottom: 20px;
            padding: 18px;
            border: 1px solid #e1e8ed;
            border-radius: 8px;
            background-color: #fafbfc;
            transition: transform 0.2s, box-shadow 0.2s;
        }}
        .news-card:hover {{
            box-shadow: 0 2px 8px rgba(0,0,0,0.08);
        }}
        .news-number {{
            font-size: 24px;
            font-weight: bold;
            color: #3498db;
            margin-right: 20px;
            min-width: 30px;
            text-align: center;
        }}
        .news-content {{
            flex-grow: 1;
        }}
        .news-content h2 {{
            margin: 0 0 8px 0;
            font-size: 18px;
        }}
        .news-content h2 a {{
            color: #2c3e50;
            text-decoration: none;
        }}
        .news-content h2 a:hover {{
            color: #3498db;
            text-decoration: underline;
        }}
        .meta {{
            font-size: 12px;
            color: #95a5a6;
            margin-bottom: 10px;
        }}
        .description {{
            font-size: 14px;
            line-height: 1.5;
            color: #555;
            margin-bottom: 10px;
        }}
        .link-bar a {{
            font-size: 13px;
            color: #3498db;
            text-decoration: none;
            font-weight: bold;
        }}
        footer {{
            text-align: center;
            margin-top: 30px;
            font-size: 12px;
            color: #bdc3c7;
        }}
    </style>
</head>
<body>
    <div class="container">
        <header>
            <h1>🎮 게임 뉴스 모음 (5선)</h1>
            <div class="header-meta">수집 일시: {timestamp_str}</div>
        </header>
        <main>
            {items_html}
        </main>
        <footer>
            Automated Game News Collector | Generated in MHTML Format
        </footer>
    </div>
</body>
</html>
"""
    return html

def create_mhtml_file(html_content, output_filepath, title):
    mhtml = MIMEMultipart("related")
    mhtml["From"] = "<GameNewsCollector@local>"
    mhtml["Subject"] = Header(title, "utf-8")
    mhtml["Date"] = time.strftime("%a, %d %b %Y %H:%M:%S %z")
    mhtml["MIME-Version"] = "1.0"

    html_part = MIMEText(html_content, "html", "utf-8")
    html_part["Content-Location"] = "file:///C:/gamenews/news.html"
    mhtml.attach(html_part)

    os.makedirs(os.path.dirname(output_filepath), exist_ok=True)
    with open(output_filepath, "wb") as f:
        f.write(mhtml.as_bytes())

def run_once():
    os.makedirs(TARGET_DIR, exist_ok=True)

    now = datetime.now()
    timestamp_str = now.strftime("%Y-%m-%d %H:%M:%S")
    file_timestamp = now.strftime("%Y%m%d_%H%M%S")

    print(f"[{timestamp_str}] Fetching 5 unique game news items...")
    news_items = collect_unique_news(count=5)

    if not news_items:
        print(f"[{timestamp_str}] No new articles found.")
        return False

    print(f"[{timestamp_str}] Successfully collected {len(news_items)} news items:")
    for idx, item in enumerate(news_items, 1):
        print(f"  {idx}. {item['title']}")

    html_content = generate_html_content(news_items, timestamp_str)
    filename = f"game_news_{file_timestamp}.mhtml"
    filepath = os.path.join(TARGET_DIR, filename)

    create_mhtml_file(html_content, filepath, f"Game News - {timestamp_str}")
    print(f"[{timestamp_str}] Saved MHTML file to: {filepath}\n")
    return True

def main():
    print("=== Game News MHTML Collector Started ===")
    print(f"Target Directory: {TARGET_DIR}\n")

    # Run once immediately
    run_once()

    # Check if run once only argument is passed (for CLI test/verification)
    if "--once" in sys.argv:
        print("Completed single run (--once mode). Exiting.")
        return

    # Hourly loop
    interval_seconds = 3600
    while True:
        print(f"Waiting {interval_seconds // 60} minutes for next run...")
        try:
            time.sleep(interval_seconds)
            run_once()
        except KeyboardInterrupt:
            print("\nProgram interrupted by user. Exiting...")
            break
        except Exception as e:
            print(f"[Error in main loop]: {e}")
            time.sleep(60)

if __name__ == "__main__":
    main()
