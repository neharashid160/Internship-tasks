"""
Task 5: Auto Tagging Support Tickets Using LLM
Objective: Automatically tag support tickets into categories using a large language model.
Dataset: Free-text Support Ticket Dataset
"""

import os
import json
import time
import pandas as pd
import numpy as np
from openai import OpenAI   # pip install openai  (works with any OpenAI-compatible API)

# ──────────────────────────────────────────────
# CONFIG
# ──────────────────────────────────────────────
# Set your API key as an environment variable:
#   export OPENAI_API_KEY="sk-..."
# Or replace the string below directly (not recommended for public repos)
API_KEY = os.getenv("OPENAI_API_KEY", "your-api-key-here")

CATEGORIES = [
    "Billing & Payment",
    "Technical Issue",
    "Account Access",
    "Shipping & Delivery",
    "Product Quality",
    "Refund & Return",
    "General Inquiry",
    "Feature Request",
]

TOP_K = 3   # Top-3 tags per ticket


# ──────────────────────────────────────────────
# SAMPLE DATASET
# ──────────────────────────────────────────────
SAMPLE_TICKETS = [
    {"id": 1, "text": "I was charged twice for my subscription this month. Please refund the extra charge."},
    {"id": 2, "text": "The app keeps crashing whenever I try to upload a photo. Running iOS 17."},
    {"id": 3, "text": "I can't log into my account. Forgot my password and reset email never arrived."},
    {"id": 4, "text": "My order #45231 has been stuck in 'processing' for 10 days. Where is it?"},
    {"id": 5, "text": "The headphones I received have a broken left ear cup. Completely unusable."},
    {"id": 6, "text": "I want to return the jacket I bought last week. It doesn't fit."},
    {"id": 7, "text": "Do you offer student discounts? I'm a university student."},
    {"id": 8, "text": "It would be great if you added dark mode to the mobile app."},
    {"id": 9, "text": "My invoice shows the wrong address. Can you update it before I pay?"},
    {"id": 10,"text": "The website is down. Getting a 503 error since this morning."},
]


# ──────────────────────────────────────────────
# PROMPTS
# ──────────────────────────────────────────────
def zero_shot_prompt(ticket_text: str) -> str:
    cats = "\n".join(f"- {c}" for c in CATEGORIES)
    return f"""You are a customer support ticket classifier.

Given the following support ticket, return the top {TOP_K} most relevant categories from the list below.
Return ONLY a valid JSON object with a single key "tags" containing a list of exactly {TOP_K} strings.

Categories:
{cats}

Ticket:
\"\"\"{ticket_text}\"\"\"

Respond with JSON only. Example: {{"tags": ["Category A", "Category B", "Category C"]}}"""


def few_shot_prompt(ticket_text: str) -> str:
    cats = "\n".join(f"- {c}" for c in CATEGORIES)
    examples = """
Examples:
Ticket: "My payment failed but money was deducted."
Output: {"tags": ["Billing & Payment", "Technical Issue", "General Inquiry"]}

Ticket: "I never received my order and need a refund."
Output: {"tags": ["Shipping & Delivery", "Refund & Return", "Billing & Payment"]}

Ticket: "The login page gives me a 404 error."
Output: {"tags": ["Account Access", "Technical Issue", "General Inquiry"]}
"""
    return f"""You are a customer support ticket classifier.

Given the following support ticket, return the top {TOP_K} most relevant categories.
Return ONLY a valid JSON object with key "tags" containing exactly {TOP_K} strings.

Categories:
{cats}
{examples}
Ticket:
\"\"\"{ticket_text}\"\"\"

Respond with JSON only."""


# ──────────────────────────────────────────────
# LLM CALLER
# ──────────────────────────────────────────────
def call_llm(prompt: str, model: str = "gpt-3.5-turbo") -> dict:
    """Call OpenAI-compatible LLM and parse JSON response."""
    client = OpenAI(api_key=API_KEY)
    response = client.chat.completions.create(
        model=model,
        messages=[{"role": "user", "content": prompt}],
        temperature=0,
        max_tokens=150,
    )
    raw = response.choices[0].message.content.strip()
    # Strip markdown fences if present
    raw = raw.replace("```json", "").replace("```", "").strip()
    return json.loads(raw)


# ──────────────────────────────────────────────
# MOCK LLM (for demo without API key)
# ──────────────────────────────────────────────
def mock_llm_tag(ticket_text: str) -> list:
    """
    Rule-based mock tagger for demo purposes.
    Replace with call_llm() when you have an API key.
    """
    text = ticket_text.lower()
    scores = {cat: 0 for cat in CATEGORIES}

    rules = {
        "Billing & Payment":   ["charge", "invoice", "payment", "bill", "refund", "subscription", "paid"],
        "Technical Issue":     ["crash", "error", "bug", "down", "503", "404", "broken", "not working", "issue"],
        "Account Access":      ["login", "password", "log in", "account", "sign in", "reset", "locked"],
        "Shipping & Delivery": ["order", "shipping", "delivery", "arrived", "tracking", "package", "stuck"],
        "Product Quality":     ["broken", "defective", "quality", "damaged", "unusable", "wrong"],
        "Refund & Return":     ["return", "refund", "exchange", "send back", "money back"],
        "General Inquiry":     ["do you", "can i", "how do", "what is", "student", "discount", "question"],
        "Feature Request":     ["would be great", "add", "feature", "dark mode", "suggest", "request"],
    }

    for cat, keywords in rules.items():
        for kw in keywords:
            if kw in text:
                scores[cat] += 1

    # Sort and return top-K
    sorted_cats = sorted(scores, key=scores.get, reverse=True)
    top = sorted_cats[:TOP_K]

    # Ensure at least one category
    if all(scores[c] == 0 for c in top):
        top = ["General Inquiry", "Technical Issue", "Billing & Payment"]

    return top


# ──────────────────────────────────────────────
# TAGGING PIPELINE
# ──────────────────────────────────────────────
def tag_tickets(tickets: list, use_api: bool = False) -> pd.DataFrame:
    """Tag all tickets with zero-shot and few-shot approaches."""
    results = []

    for ticket in tickets:
        print(f"  Processing ticket #{ticket['id']} ...", end=" ")

        if use_api:
            try:
                zs_tags = call_llm(zero_shot_prompt(ticket["text"]))["tags"]
                time.sleep(0.5)   # rate limit
                fs_tags = call_llm(few_shot_prompt(ticket["text"]))["tags"]
            except Exception as e:
                print(f"API error: {e}. Falling back to mock.")
                zs_tags = mock_llm_tag(ticket["text"])
                fs_tags = mock_llm_tag(ticket["text"])
        else:
            zs_tags = mock_llm_tag(ticket["text"])
            fs_tags = mock_llm_tag(ticket["text"])   # same mock for demo

        results.append({
            "ticket_id":         ticket["id"],
            "ticket_text":       ticket["text"],
            "zero_shot_tags":    zs_tags,
            "few_shot_tags":     fs_tags,
            "zero_shot_top1":    zs_tags[0] if zs_tags else "",
            "few_shot_top1":     fs_tags[0] if fs_tags else "",
            "tags_match":        zs_tags == fs_tags,
        })
        print("✅")

    return pd.DataFrame(results)


# ──────────────────────────────────────────────
# EVALUATION (when ground truth available)
# ──────────────────────────────────────────────
def evaluate_accuracy(df: pd.DataFrame, ground_truth_col: str = None) -> None:
    """Compare zero-shot vs few-shot if ground truth available."""
    if ground_truth_col and ground_truth_col in df.columns:
        zs_acc = (df["zero_shot_top1"] == df[ground_truth_col]).mean()
        fs_acc = (df["few_shot_top1"]  == df[ground_truth_col]).mean()
        print(f"\n📊 Zero-shot accuracy : {zs_acc:.2%}")
        print(f"📊 Few-shot  accuracy : {fs_acc:.2%}")
    else:
        match_rate = df["tags_match"].mean()
        print(f"\n📊 Zero-shot vs Few-shot agreement rate : {match_rate:.2%}")


# ──────────────────────────────────────────────
# MAIN
# ──────────────────────────────────────────────
def main():
    print("=" * 60)
    print("  Task 5: Auto-Tagging Support Tickets Using LLM")
    print("=" * 60)

    # Detect whether API key is set
    use_api = API_KEY not in ("your-api-key-here", "", None)
    if use_api:
        print("🔑 API key detected — using real LLM calls.")
    else:
        print("⚙️  No API key — using rule-based mock tagger (demo mode).")
        print("   Set OPENAI_API_KEY env var to use actual GPT.")

    print(f"\n📋 Tagging {len(SAMPLE_TICKETS)} support tickets ...\n")
    df = tag_tickets(SAMPLE_TICKETS, use_api=use_api)

    # Display results
    print("\n\n" + "=" * 60)
    print("  RESULTS")
    print("=" * 60)
    for _, row in df.iterrows():
        print(f"\nTicket #{row['ticket_id']}:")
        print(f"  Text       : {row['ticket_text'][:80]}...")
        print(f"  Zero-shot  : {row['zero_shot_tags']}")
        print(f"  Few-shot   : {row['few_shot_tags']}")
        print(f"  Match      : {'✅' if row['tags_match'] else '❌'}")

    evaluate_accuracy(df)

    # Save
    df.to_csv("tagged_tickets.csv", index=False)
    print("\n\n✅ Results saved to: tagged_tickets.csv")
    print("✅ Task 5 Complete!")


if __name__ == "__main__":
    main()
