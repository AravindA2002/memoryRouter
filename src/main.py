from __future__ import annotations
import time
from dotenv import load_dotenv

from router.memory_router import MemoryRouter
from router.embeddings_openai import OpenAIEmbedder

load_dotenv() 

def seed_memories(r: MemoryRouter):
    print("\nStoring pre defined memories...")
    r.store_memory("Recharge for vodafone number ending 080 by today", "short_term",
                   {"user_id": "u1"}, ttl_seconds=30)   
    r.store_memory("Need to completer the wrapper memory router for the memory system that we are building", "working",
                   {"scope": "sess42", "user_id": "u1"}, ttl_seconds=120)
    '''
    r.store_memory("Got Masters Degree from Arizona State University", "long_term",
                   {"user_id": "u1"})
    r.store_memory("I have allergy with pineapples", "semantic",
                   {"domain": "ops", "user_id": "u1"})'''
    print("Done\n")

def do_retrieve(r: MemoryRouter, query: str, where: str = "auto", top_k: int = 5, scope: str | None = None):
    print(f"\n[RETRIEVE] Q: {query}  (where={where}, top_k={top_k}, scope={scope})")
    hits = r.retrieve(query, where=where, top_k=top_k, working_scope=scope)
    if not hits:
        print("  No results.\n")
        return
    for m, score in hits:
        print(f"  score={score:.3f}, type={m.memory_type}, text={m.text}")
    print()

def run_retrieval_demos(r: MemoryRouter):
    demos = [
        ("what degree did the user get from ASU", "auto", None),
        ("what are my allergy foods", "auto", None),
        ("when should i recharge ?", "auto", "sess42"),  
    ]
    for q, where, scope in demos:
        do_retrieve(r, q, where=where, top_k=5, scope=scope)


def main():
    r = MemoryRouter(OpenAIEmbedder())

    print("Choose an option:")
    print("1.Store pre defined memories")
    print("2.Retrieve - run stored queries)")
    print("3.Retrieve")
    print("4.Exit")

    choice = input().strip()

    if choice == "1":
        seed_memories(r)

    elif choice == "2":
        run_retrieval_demos(r)

    elif choice == "3":
        q = input("Enter your query: ").strip()
        if not q:
            print("No Querry Exiting.")
            return
        print("Search where?")
        where = input("where = ").strip() or "auto"
        scope = None
        if where in ("working", "auto"):
            scope = input("working scope: ").strip() or None
        try:
            top_k = int(input("top 5: ").strip() or "5")
        except ValueError:
            top_k = 5
        do_retrieve(r, q, where=where, top_k=top_k, scope=scope)

    elif choice == "4":
        print("Exiting")
        return
    else:
        print("Exiting")

if __name__ == "__main__":
    main()
