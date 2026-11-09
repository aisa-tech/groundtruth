# src/generate.py
import json, anthropic
client = anthropic.Anthropic()  # ANTHROPIC_API_KEY ortam değişkeninden okunur
MODEL = "claude-sonnet-5-5"

def answer(query, chunks):
    ctx = "\n\n".join(f"[{i+1}] {c['text']}" for i, c in enumerate(chunks[:5]))
    msg = client.messages.create(model=MODEL, max_tokens=400, messages=[{
        "role": "user",
        "content": f"Answer only from the passages. Cite like [1]. "
                   f"If they do not contain the answer, say so.\n\n{ctx}\n\nQuestion: {query}"}])
    return msg.content[0].text

def judge(query, chunks, ans):
    ctx = "\n\n".join(c["text"] for c in chunks[:5])
    prompt = ("Score two things from 1 to 5 and reply as JSON "
              '{"faithfulness": n, "relevance": n, "reason": "..."}.\n'
              "faithfulness: is every claim in the answer supported by the passages?\n"
              "relevance: does the answer address the question?\n\n"
              f"Passages:\n{ctx}\n\nQuestion: {query}\n\nAnswer: {ans}")
    msg = client.messages.create(model=MODEL, max_tokens=300,
                                 messages=[{"role": "user", "content": prompt}])
    return json.loads(msg.content[0].text)