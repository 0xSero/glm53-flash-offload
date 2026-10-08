#!/usr/bin/env python3
"""N134 capture prompt set: code, chat, reasoning, writing, knowledge, multilingual, structured output, long context.
Writes prompts.jsonl: {"id", "cat", "thinking", "messages" | "long": {"files": [...], "ask": ...}}.
Long-context items are assembled on the capture host from local source files (see cap_client.py)."""
import json, random, sys

CODE = [
    "Write a Python function that parses an ISO-8601 duration string like 'P3DT4H12M' into seconds, with tests.",
    "Implement an LRU cache in Rust with O(1) get and put, generic over key and value. Explain the ownership choices.",
    "Write a C function that reverses a singly linked list in place, then one that detects a cycle. Include a small main().",
    "In TypeScript, write a debounce and a throttle utility with correct `this` handling and cancellation.",
    "Explain what this Python does and find the bug:\n\n```python\ndef chunks(xs, n):\n    return [xs[i:i+n] for i in range(0, len(xs)-n, n)]\n```",
    "Write a SQL query that returns, for each customer, their three most recent orders and the running total of spend. Assume tables customers(id, name) and orders(id, customer_id, created_at, amount).",
    "Write a bash script that finds the 10 largest files under a directory, skipping .git folders, and prints human-readable sizes.",
    "Implement Dijkstra's algorithm in Go using container/heap, and show it on a small graph.",
    "Write a CUDA kernel that computes a row-wise softmax over a [rows, cols] float matrix, one warp per row. Explain the numerics.",
    "Write a Python asyncio web crawler that respects a concurrency limit of 8 and a per-host delay, using aiohttp.",
    "Refactor this JavaScript into modern idiomatic code:\n\n```js\nvar self = this; function load(cb){ $.get('/api', function(d){ if(d && d.items){ for(var i=0;i<d.items.length;i++){ self.add(d.items[i]); } cb(null); } else { cb('no data'); } }); }\n```",
    "Write a Haskell function that computes all permutations of a list, then explain its complexity.",
    "Implement a thread-safe bounded blocking queue in Java without using java.util.concurrent classes.",
    "Write a regex that matches valid IPv4 addresses (0-255 per octet) and explain each part. Then write it as a non-regex Python validator.",
    "Design a REST API for a small library system (books, members, loans). Give endpoints, payloads and error codes.",
    "Write a Dockerfile for a Python FastAPI app with a multi-stage build, non-root user and a healthcheck.",
    "Explain the difference between a mutex and a semaphore with a C++ example of each, and when to use a condition variable.",
    "Write a Python script that reads a large CSV in chunks with pandas and computes per-group medians without loading it all.",
    "Implement a trie in Python with insert, search, starts_with and delete, plus unit tests with pytest.",
    "Write a Kotlin data class for a user profile and a function that validates email and age, returning a sealed result type.",
    "Write a minimal Makefile for a C project with src/ and include/ directories, automatic dependency generation, and debug/release targets.",
    "Implement the A* search algorithm on a 2D grid with obstacles in Python and print the path.",
    "Write a React component for a searchable, paginated table that fetches from /api/items?query=&page=. Use hooks.",
    "Explain how Python's GIL affects multithreaded CPU-bound code and show a multiprocessing alternative with timings.",
    "Write a function in C++20 that splits a string_view on a delimiter and returns a vector of string_views. Discuss lifetime pitfalls.",
    "Write a Python decorator that retries a function with exponential backoff and jitter, configurable exceptions and max attempts.",
    "Implement a simple tokenizer and recursive-descent parser for arithmetic expressions with + - * / and parentheses, in Python.",
    "Write a Rust program that reads lines from stdin, counts word frequencies, and prints the top 20 sorted by count.",
    "Explain what a memory leak looks like in a long-running Node.js server and how to find it with heap snapshots.",
    "Write a PyTorch training loop for a small MLP on MNIST with mixed precision and gradient clipping.",
]
CHAT = [
    "I've been feeling unmotivated at work lately. Any practical suggestions?",
    "What's a good way to start learning the piano as an adult with about 20 minutes a day?",
    "Can you help me plan a 4-day trip to Lisbon on a moderate budget?",
    "My sourdough keeps coming out flat. What could be going wrong?",
    "How do I politely decline a friend's wedding invitation?",
    "What are some good habits for keeping a small apartment tidy?",
    "I'm adopting a cat for the first time. What should I prepare?",
    "Give me a weekly vegetarian meal plan that is high in protein.",
    "What should I consider when choosing between renting and buying a home?",
    "How can I get better at public speaking?",
    "Recommend some science fiction novels for someone who loved The Martian.",
    "How do I explain to my kids why the sky is blue?",
    "What's the best way to prepare for a technical job interview in two weeks?",
    "I want to start running. Can you give me a beginner 8-week plan?",
    "Tell me a fun fact about octopuses and then explain why it's true.",
    "How do I negotiate a higher salary in a job offer?",
    "What are the pros and cons of electric cars today?",
    "Help me write a birthday message for my grandmother who is turning 90.",
    "How do I deal with a noisy upstairs neighbor?",
    "What are some low-cost hobbies I could pick up this winter?",
]
REASON = [
    "A bat and a ball cost $1.10 in total. The bat costs $1.00 more than the ball. How much does the ball cost? Explain carefully.",
    "Prove that the square root of 2 is irrational.",
    "How many trailing zeros does 1000! have? Show the reasoning.",
    "Three people check into a hotel room that costs $30. They each pay $10. The clerk realizes the room is $25 and gives $5 to the bellboy, who keeps $2 and returns $1 to each. Where is the missing dollar?",
    "Find all integer solutions to x^2 - y^2 = 45.",
    "A snail climbs 3 meters up a 20 meter well each day and slides back 2 meters each night. On which day does it get out?",
    "Solve: the sum of three consecutive odd integers is 87. Find them, then generalize to n consecutive odd integers summing to S.",
    "What is the probability that in a group of 23 people at least two share a birthday? Derive it.",
    "Prove by induction that 1^3 + 2^3 + ... + n^3 = (n(n+1)/2)^2.",
    "You have 12 coins, one of which is counterfeit and either heavier or lighter. With a balance scale and three weighings, how do you find it and tell whether it is heavier or lighter?",
    "Compute the integral of x * e^(2x) dx and verify by differentiation.",
    "A train leaves city A at 60 km/h. Two hours later another leaves A on the same track at 90 km/h. When and where does the second catch the first?",
    "Is 2^61 - 1 prime? Explain how you'd check, and what is known about it.",
    "In how many ways can 8 rooks be placed on a chessboard so that none attack each other? And 8 queens? Explain the difference.",
    "Explain the Monty Hall problem and prove the switching strategy wins 2/3 of the time.",
    "Find the eigenvalues and eigenvectors of the matrix [[2, 1], [1, 2]] and explain their geometric meaning.",
    "A rectangle has perimeter 40. What dimensions maximize its area? Prove it with and without calculus.",
    "Knights always tell the truth and knaves always lie. A says: 'B is a knave.' B says: 'A and I are of the same kind.' What are A and B?",
    "What is the expected number of rolls of a fair die to see all six faces? Derive it.",
    "Show that there are infinitely many primes, then explain why there are arbitrarily long gaps between consecutive primes.",
    "Let f(x) = x^3 - 6x^2 + 9x + 1. Find the local extrema and inflection points and sketch the behavior.",
    "A password has 8 characters from 62 symbols. How long would a brute force take at 10^10 guesses per second? Compare with 12 characters.",
    "If you fold a sheet of paper 0.1 mm thick in half 42 times, how thick would it be? Is it physically possible?",
    "Two players alternately take 1 to 3 stones from a pile of 21. The player who takes the last stone loses. Who wins with perfect play?",
    "Estimate how many piano tuners work in Chicago. Show a Fermi estimate with stated assumptions.",
    "Explain why 0.999... equals 1 using three different arguments.",
    "Given a 3x3 magic square using 1..9, prove the center must be 5.",
    "A drawer has 10 black and 10 white socks. In the dark, how many must you take to be sure of a matching pair? Of a black pair?",
    "Solve the recurrence T(n) = 2T(n/2) + n log n and explain which case of the master theorem applies, if any.",
    "Prove that every tree with n vertices has exactly n-1 edges.",
]
WRITE = [
    "Write a short story (about 600 words) about a lighthouse keeper who finds a message in a bottle from the future.",
    "Write a persuasive essay arguing for a four-day work week, with counterarguments addressed.",
    "Write a poem about autumn in the style of Mary Oliver.",
    "Write a product description for a handmade ceramic coffee mug, warm and concise.",
    "Draft a polite but firm email to a contractor who is three weeks behind schedule.",
    "Write a children's bedtime story about a dragon who is afraid of the dark.",
    "Write a cover letter for a junior data analyst position, for someone switching from teaching.",
    "Summarize the plot of Hamlet in 200 words, then give three themes with one quote each.",
    "Write a dialogue between Socrates and a modern software engineer about whether AI can know things.",
    "Write a eulogy for a beloved family dog named Biscuit.",
    "Write the opening chapter of a mystery novel set in a 1920s Istanbul hotel.",
    "Write a limerick about a programmer who forgot a semicolon, then a sonnet on the same theme.",
    "Write a blog post explaining compound interest to teenagers.",
    "Write a speech for a high school graduation about failure and curiosity.",
    "Rewrite this sentence in five different tones (formal, casual, sarcastic, poetic, legal): 'The meeting is moved to Thursday.'",
]
KNOW = [
    "Explain how the transformer architecture works, from tokens to logits, for someone who knows basic linear algebra.",
    "What caused the 2008 financial crisis? Give a structured explanation.",
    "How does the immune system distinguish self from non-self?",
    "Explain quantum entanglement without misleading analogies.",
    "What were the main causes and consequences of the fall of the Western Roman Empire?",
    "How do mixture-of-experts language models route tokens, and what are the trade-offs of top-k routing?",
    "How does a refrigerator work? Explain the thermodynamic cycle.",
    "Explain the difference between TCP and QUIC and why HTTP/3 uses QUIC.",
    "What is CRISPR-Cas9 and how is it used to edit genes?",
    "Explain how public-key cryptography works, with RSA as the example, including a toy calculation.",
    "Why do we have seasons on Earth? Address the common misconception about distance from the Sun.",
    "Explain the history and significance of the Silk Road.",
    "How do SSDs store data, and why do they wear out? Explain NAND, FTL and write amplification.",
    "What is the difference between weather and climate, and how do climate models work?",
    "Explain how a CPU cache hierarchy works and why data layout matters for performance.",
    "Describe the main schools of thought in moral philosophy and how they would handle the trolley problem.",
    "How does GPS achieve meter-level accuracy, including relativistic corrections?",
    "Explain the causes of inflation and the tools central banks use against it.",
    "What is the Riemann hypothesis and why does it matter?",
    "Explain how vaccines based on mRNA differ from traditional vaccines.",
]
MULTI = [
    ("zh", "请用中文解释一下什么是机器学习中的过拟合，以及如何防止它。"),
    ("zh", "写一首关于春天的七言绝句，并解释其中的意象。"),
    ("zh", "帮我写一个Python函数，计算两个日期之间的工作日数量，并加上中文注释。"),
    ("zh", "请比较一下唐朝和宋朝在经济和文化上的主要区别。"),
    ("es", "Explica en español cómo funciona la fotosíntesis y por qué es importante para la vida en la Tierra."),
    ("es", "Escribe un cuento corto sobre un gato que quería ser astronauta."),
    ("fr", "Explique en français les causes de la Révolution française et ses conséquences principales."),
    ("fr", "Rédige une lettre de motivation pour un poste de développeur web junior."),
    ("de", "Erkläre auf Deutsch den Unterschied zwischen Wetter und Klima und warum das wichtig ist."),
    ("de", "Schreibe eine kurze Geschichte über einen Roboter, der Gärtner werden möchte."),
    ("ja", "日本語で、量子コンピュータの基本的な仕組みをわかりやすく説明してください。"),
    ("ja", "東京で3日間観光する旅行プランを立ててください。"),
    ("ko", "한국어로 인공지능의 윤리적 문제에 대해 설명해 주세요."),
    ("ru", "Объясни по-русски, как работает блокчейн и где он реально полезен."),
    ("pt", "Explique em português como funciona o sistema imunológico humano."),
    ("ar", "اشرح باللغة العربية أهمية الطاقة المتجددة ومستقبلها."),
]
STRUCT = [
    "Return a JSON object describing three fictional employees with fields name, role, skills (array), and manager (nullable). Only JSON.",
    "Convert this to a Markdown table: apples 3 red, bananas 12 yellow, grapes 40 purple, limes 7 green. Add a total row.",
    "Extract all dates, people and organizations from this text as JSON: 'On March 3, 2021, Alice Chen joined Acme Corp after leaving Globex on February 1. Bob Ruiz from Initech met her on April 15.'",
    "Write a YAML GitHub Actions workflow that runs pytest on Python 3.10-3.12 and uploads coverage.",
    "Generate a JSON Schema for an order object with id, customer, items (sku, qty, price), and status enum.",
    "Write a Mermaid sequence diagram for an OAuth 2.0 authorization code flow with PKCE.",
    "Produce a CSV with 15 rows of plausible weather data for a week in two cities, with columns date, city, high_c, low_c, precip_mm.",
    "Write an OpenAPI 3.0 snippet for a POST /users endpoint with validation rules and error responses.",
]
MULTITURN = [
    [{"role": "user", "content": "I'm building a CLI todo app in Python. Suggest a file format for storing tasks."},
     {"role": "assistant", "content": "A simple JSON file works well: a list of objects with id, title, done, created_at and optional due date."},
     {"role": "user", "content": "Good. Now write the full app with add, list, done and delete commands using argparse."}],
    [{"role": "user", "content": "What's the capital of Canada?"},
     {"role": "assistant", "content": "Ottawa."},
     {"role": "user", "content": "Why not Toronto or Montreal? Give the history."}],
    [{"role": "user", "content": "Explain gradient descent."},
     {"role": "assistant", "content": "Gradient descent iteratively moves parameters in the direction of the negative gradient of the loss to reduce it."},
     {"role": "user", "content": "Now explain Adam and why it often trains faster, with the update equations."}],
    [{"role": "system", "content": "You are a concise senior SRE. Answer with concrete commands."},
     {"role": "user", "content": "A Linux server's disk is 100% full and the app is down. Walk me through triage."}],
    [{"role": "system", "content": "You are a patient math tutor. Ask guiding questions instead of giving answers outright."},
     {"role": "user", "content": "I don't get why the derivative of x^2 is 2x."}],
    [{"role": "user", "content": "Let's play 20 questions. I'm thinking of an animal. Ask your first question and keep a running list of what you know."}],
]
# long context: (files on the capture host relative to the code dir, ask)
LONG = [
    (["nv2_host.cpp"], "Review this C++ file. Summarize its architecture, then list the five riskiest concurrency issues you can find with line references."),
    (["nv2.py", "expert_cache.py"], "Explain how these two Python modules cooperate during decode. Then propose three simplifications."),
    (["nv2_dev.cu"], "Explain what each CUDA kernel in this file does and where the synchronization points between device and host are."),
    (["serve.py"], "Document this server's HTTP API as a reference table, then point out any input-validation gaps."),
    (["cpu_tier.py", "nv_tier.py"], "Compare these two modules: what does each do, what do they share, and how would you unify them?"),
    (["ft_core.h"], "Explain the numerical kernels in this header and how the AVX2 code paths are organized."),
    (["k_hcfuse.py", "k_overlap.py", "triton_pin.py"], "Summarize each of these files in one paragraph and describe how they would be tested."),
    (["expert_cache.py"], "Write unit tests (pytest) for the CLOCK cache logic in this module. Mock out CUDA where needed."),
]


def main(out):
    rows = []
    rng = random.Random(134)
    add = lambda cat, thinking, msgs: rows.append({"cat": cat, "thinking": thinking, "messages": msgs})
    for i, p in enumerate(CODE):
        add("code", i % 4 != 3, [{"role": "user", "content": p}])
    for i, p in enumerate(CHAT):
        add("chat", i % 2 == 0, [{"role": "user", "content": p}])
    for p in REASON:
        add("reasoning", True, [{"role": "user", "content": p}])
    for i, p in enumerate(WRITE):
        add("writing", i % 3 == 0, [{"role": "user", "content": p}])
    for i, p in enumerate(KNOW):
        add("knowledge", i % 2 == 1, [{"role": "user", "content": p}])
    for i, (lang, p) in enumerate(MULTI):
        add("multilingual-" + lang, i % 2 == 0, [{"role": "user", "content": p}])
    for i, p in enumerate(STRUCT):
        add("structured", i % 2 == 0, [{"role": "user", "content": p}])
    for m in MULTITURN:
        add("multiturn", True, m)
    for i, (files, ask) in enumerate(LONG):
        rows.append({"cat": "longctx", "thinking": i % 2 == 0, "long": {"files": files, "ask": ask}})
    rng.shuffle(rows)
    for i, r in enumerate(rows):
        r["id"] = f"n134-{i:03d}"
    with open(out, "w") as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
    cats = {}
    for r in rows:
        cats[r["cat"].split("-")[0]] = cats.get(r["cat"].split("-")[0], 0) + 1
    print(len(rows), cats)


if __name__ == "__main__":
    main(sys.argv[1] if len(sys.argv) > 1 else "prompts.jsonl")
