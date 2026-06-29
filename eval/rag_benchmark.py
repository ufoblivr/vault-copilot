"""
RAG Evaluation Benchmark — 50 synthetic receipt Q&A pairs.
Used to measure retrieval quality (recall@k, MRR, precision).
"""
from dataclasses import dataclass, field
from typing import List


@dataclass
class BenchmarkEntry:
    """A single evaluation entry."""
    query: str
    relevant_doc_ids: List[str]
    category: str  # item_search, amount, store, date, general


# ---------------------------------------------------------------------------
# Synthetic receipt corpus — 25 diverse receipts
# ---------------------------------------------------------------------------
SYNTHETIC_CORPUS = [
    {"id": "1", "text": "Spent $45.99 at Whole Foods on 2024-01-15. Items: organic milk, avocados, quinoa, almond butter.", "metadata": {"store": "Whole Foods"}},
    {"id": "2", "text": "Spent $12.50 at Starbucks on 2024-01-16. Items: grande latte, blueberry croissant.", "metadata": {"store": "Starbucks"}},
    {"id": "3", "text": "Spent $199.99 at Amazon on 2024-01-20. Items: Sony WH-1000XM5 wireless headphones.", "metadata": {"store": "Amazon"}},
    {"id": "4", "text": "Spent $67.40 at Target on 2024-02-01. Items: paper towels, cereal, laundry detergent, dish soap.", "metadata": {"store": "Target"}},
    {"id": "5", "text": "Spent $55.00 at Shell Gas on 2024-02-10. Items: premium unleaded fuel, windshield wiper fluid.", "metadata": {"store": "Shell Gas"}},
    {"id": "6", "text": "Spent $23.75 at Chipotle on 2024-02-12. Items: chicken burrito bowl, guacamole, large drink.", "metadata": {"store": "Chipotle"}},
    {"id": "7", "text": "Spent $89.99 at Costco on 2024-02-15. Items: bulk toilet paper, rotisserie chicken, olive oil, mixed nuts.", "metadata": {"store": "Costco"}},
    {"id": "8", "text": "Spent $15.99 at Netflix on 2024-02-01. Items: monthly streaming subscription.", "metadata": {"store": "Netflix"}},
    {"id": "9", "text": "Spent $34.50 at Walmart on 2024-02-20. Items: ground beef, rice, canned beans, tortillas, cheese.", "metadata": {"store": "Walmart"}},
    {"id": "10", "text": "Spent $250.00 at Best Buy on 2024-03-01. Items: Samsung 27-inch 4K monitor.", "metadata": {"store": "Best Buy"}},
    {"id": "11", "text": "Spent $8.99 at Spotify on 2024-03-01. Items: premium music subscription.", "metadata": {"store": "Spotify"}},
    {"id": "12", "text": "Spent $42.30 at Trader Joes on 2024-03-05. Items: frozen orange chicken, cauliflower gnocchi, dark chocolate almonds.", "metadata": {"store": "Trader Joes"}},
    {"id": "13", "text": "Spent $112.00 at CVS Pharmacy on 2024-03-10. Items: prescription medication, vitamin D supplements, hand sanitizer.", "metadata": {"store": "CVS Pharmacy"}},
    {"id": "14", "text": "Spent $18.00 at Uber Eats on 2024-03-12. Items: pad thai delivery, spring rolls.", "metadata": {"store": "Uber Eats"}},
    {"id": "15", "text": "Spent $75.00 at Nike on 2024-03-15. Items: running shoes Air Zoom Pegasus.", "metadata": {"store": "Nike"}},
    {"id": "16", "text": "Spent $31.20 at Whole Foods on 2024-03-18. Items: salmon fillet, asparagus, brown rice, lemons.", "metadata": {"store": "Whole Foods"}},
    {"id": "17", "text": "Spent $9.50 at Starbucks on 2024-03-19. Items: iced americano, banana bread.", "metadata": {"store": "Starbucks"}},
    {"id": "18", "text": "Spent $450.00 at Apple Store on 2024-03-22. Items: AirPods Pro 2nd generation, AppleCare warranty.", "metadata": {"store": "Apple Store"}},
    {"id": "19", "text": "Spent $28.99 at Dominos on 2024-03-25. Items: large pepperoni pizza, garlic bread, 2-liter Coke.", "metadata": {"store": "Dominos"}},
    {"id": "20", "text": "Spent $62.50 at Shell Gas on 2024-03-28. Items: regular unleaded fuel.", "metadata": {"store": "Shell Gas"}},
    {"id": "21", "text": "Spent $15.99 at Netflix on 2024-03-01. Items: monthly streaming subscription.", "metadata": {"store": "Netflix"}},
    {"id": "22", "text": "Spent $95.00 at Home Depot on 2024-04-01. Items: power drill, wood screws, sandpaper, paint brush.", "metadata": {"store": "Home Depot"}},
    {"id": "23", "text": "Spent $37.80 at Whole Foods on 2024-04-05. Items: organic eggs, sourdough bread, kale, blueberries.", "metadata": {"store": "Whole Foods"}},
    {"id": "24", "text": "Spent $8.99 at Spotify on 2024-04-01. Items: premium music subscription.", "metadata": {"store": "Spotify"}},
    {"id": "25", "text": "Spent $145.00 at Costco on 2024-04-10. Items: 65-inch TV mount, HDMI cables, surge protector.", "metadata": {"store": "Costco"}},
]


# ---------------------------------------------------------------------------
# 50 benchmark queries across 5 categories
# ---------------------------------------------------------------------------
BENCHMARK_QUERIES = [
    # === item_search (15 queries) ===
    BenchmarkEntry("Where did I buy avocados?", ["1"], "item_search"),
    BenchmarkEntry("Did I purchase any headphones?", ["3"], "item_search"),
    BenchmarkEntry("Which store had organic milk?", ["1"], "item_search"),
    BenchmarkEntry("Where did I get a burrito bowl?", ["6"], "item_search"),
    BenchmarkEntry("Did I buy any salmon?", ["16"], "item_search"),
    BenchmarkEntry("Where did I purchase running shoes?", ["15"], "item_search"),
    BenchmarkEntry("Which receipt has orange chicken?", ["12"], "item_search"),
    BenchmarkEntry("Did I buy any pizza recently?", ["19"], "item_search"),
    BenchmarkEntry("Where did I get AirPods?", ["18"], "item_search"),
    BenchmarkEntry("Which store sold me a power drill?", ["22"], "item_search"),
    BenchmarkEntry("Did I buy vitamin supplements anywhere?", ["13"], "item_search"),
    BenchmarkEntry("Where did I get laundry detergent?", ["4"], "item_search"),
    BenchmarkEntry("Did I purchase any olive oil?", ["7"], "item_search"),
    BenchmarkEntry("Which receipt has blueberries?", ["23"], "item_search"),
    BenchmarkEntry("Where did I buy a monitor?", ["10"], "item_search"),

    # === store (10 queries) ===
    BenchmarkEntry("How much did I spend at Starbucks?", ["2", "17"], "store"),
    BenchmarkEntry("What did I buy from Amazon?", ["3"], "store"),
    BenchmarkEntry("Show me all Whole Foods purchases.", ["1", "16", "23"], "store"),
    BenchmarkEntry("How much did I spend at Costco?", ["7", "25"], "store"),
    BenchmarkEntry("What did I get from Target?", ["4"], "store"),
    BenchmarkEntry("Show me my Shell Gas receipts.", ["5", "20"], "store"),
    BenchmarkEntry("How much did I spend at the Apple Store?", ["18"], "store"),
    BenchmarkEntry("What did I order from Uber Eats?", ["14"], "store"),
    BenchmarkEntry("Show me my CVS purchases.", ["13"], "store"),
    BenchmarkEntry("How much did I spend at Home Depot?", ["22"], "store"),

    # === date (10 queries) ===
    BenchmarkEntry("What did I buy on January 15th?", ["1"], "date"),
    BenchmarkEntry("Show me purchases from February.", ["4", "5", "6", "7", "8", "9"], "date"),
    BenchmarkEntry("What did I spend on March 1st?", ["10", "11", "21"], "date"),
    BenchmarkEntry("Any purchases in April?", ["22", "23", "24", "25"], "date"),
    BenchmarkEntry("What did I buy on 2024-01-20?", ["3"], "date"),
    BenchmarkEntry("Show me my March 22 receipt.", ["18"], "date"),
    BenchmarkEntry("What did I buy on February 15th?", ["7"], "date"),
    BenchmarkEntry("Purchases from January 16?", ["2"], "date"),
    BenchmarkEntry("What happened on March 25th?", ["19"], "date"),
    BenchmarkEntry("Show me receipts from February 20.", ["9"], "date"),

    # === amount (10 queries) ===
    BenchmarkEntry("What was my most expensive purchase?", ["18"], "amount"),
    BenchmarkEntry("Did I spend over $200 anywhere?", ["3", "10", "18"], "amount"),
    BenchmarkEntry("What purchases were under $15?", ["2", "8", "11", "17", "24"], "amount"),
    BenchmarkEntry("How much was the cheapest thing I bought?", ["8", "11", "24"], "amount"),
    BenchmarkEntry("Any purchases over $100?", ["3", "10", "13", "18", "25"], "amount"),
    BenchmarkEntry("What did I spend $75 on?", ["15"], "amount"),
    BenchmarkEntry("Purchases around $30?", ["9", "12", "16", "19"], "amount"),
    BenchmarkEntry("Most expensive electronics purchase?", ["18"], "amount"),
    BenchmarkEntry("What cost $250?", ["10"], "amount"),
    BenchmarkEntry("Cheapest food purchase?", ["2", "17"], "amount"),

    # === general (5 queries) ===
    BenchmarkEntry("What groceries did I buy?", ["1", "4", "7", "9", "12", "16", "23"], "general"),
    BenchmarkEntry("Any coffee purchases?", ["2", "17"], "general"),
    BenchmarkEntry("Show me all my subscriptions.", ["8", "11", "21", "24"], "general"),
    BenchmarkEntry("What food delivery orders did I make?", ["14", "19"], "general"),
    BenchmarkEntry("Any gas station visits?", ["5", "20"], "general"),
]
