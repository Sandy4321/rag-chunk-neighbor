import os, glob, re, json
import numpy as np
from collections import Counter
from datetime import datetime
from dotenv import load_dotenv

from langchain_community.document_loaders import PyMuPDFLoader
from langchain_openai import OpenAIEmbeddings, ChatOpenAI
from langchain.vectorstores.faiss import FAISS

# === CONFIG ===
DATA_DIR = "mydata"
ENV_PATH = r"e:api_keys.env"
MODEL_NAME = "gpt-4o-mini"
PDF_DENSE_N1, PDF_DENSE_N2 = 2, 2
PDF_SPARSE_N1, PDF_SPARSE_N2 = 1, 1
CHAT_DENSE_N1, CHAT_DENSE_N2 = 1, 1
CHAT_SPARSE_N1, CHAT_SPARSE_N2 = 1, 1
TOP_K_DENSE = 5
TOP_K_SPARSE = 4

LOG_DIR = "logs"
MIN_NGRAMS = 3
NGRAM_N = 4
MAX_CHAT_HISTORY = 30

PROMPT_TMPL = (
    "You are a helpful AI assistant leveraging Retrieval‑Augmented Generation (RAG).\n"
    "Each PDF file is a **separate story** — keep details distinct in your answer.\n"
    "Turn #: {turn}\n\n"
    "### Retrieved Context ###\n{context}\n\n"
    "### Recent Chat History ###\n{history}\n\n"
    "### Question ###\n{question}\n\n"
    "### Answer ###\n"
)

os.makedirs(LOG_DIR, exist_ok=True)
load_dotenv(ENV_PATH, override=True)
embeddings = OpenAIEmbeddings()
llm = ChatOpenAI(model=MODEL_NAME, temperature=0.1)

def extract_tokens(text, n=NGRAM_N):
    tokens = []
    letters = "".join(c for c in text if c.isalpha())
    for i in range(len(letters)-n+1):
        ng = letters[i:i+n].lower()
        if len(ng) == n:
            tokens.append(ng)
    # Numbers (any digit sequence)
    for m in re.finditer(r"\d+", text):
        tokens.append(m.group(0))
    # Number+letters tokens (e.g. 12abc)
    for m in re.finditer(r"\b\d+[a-zA-Z]+\b", text):
        tokens.append(m.group(0).lower())
        tokens.append(re.match(r"\d+", m.group(0)).group(0))
    # Capitalized words as tokens (for names/abbrs, any length, always lower)
    for w in re.findall(r"\b[A-Z][a-zA-Z]*\b", text):
        tokens.append(w.lower())
    return tokens

def short_summary(text, num_sentences=3):
    stopwords = set(["a", "about", "above", "after", "again", "against", "all", "am", "an", "and", "any", "are", "as", "at", "be", "because", "been", "before", "being", "below", "between", "both", "but", "by", "can", "did", "do", "does", "doing", "don", "down", "during", "each", "few", "for", "from", "further", "had", "has", "have", "having", "he", "her", "here", "hers", "herself", "him", "himself", "his", "how", "i", "if", "in", "into", "is", "it", "its", "itself", "just", "me", "more", "most", "my", "myself", "no", "nor", "not", "now", "o", "of", "off", "on", "once", "only", "or", "other", "our", "ours", "ourselves", "out", "over", "own", "s", "same", "she", "should", "so", "some", "such", "t", "than", "that", "the", "their", "theirs", "them", "themselves", "then", "there", "these", "they", "this", "those", "through", "to", "too", "under", "until", "up", "very", "was", "we", "were", "what", "when", "where", "which", "while", "who", "whom", "why", "will", "with", "you", "your", "yours", "yourself", "yourselves"])
    sentences = [s.strip() for s in re.split(r"(?<=[.!?])\s+", text) if s.strip()]
    if not sentences: return ""
    words = re.findall(r'\b\w+\b', text.lower())
    word_freq = Counter(w for w in words if w not in stopwords)
    if not word_freq: return sentences[0]
    sentence_scores = {}
    for i, sentence in enumerate(sentences):
        sentence_words = re.findall(r'\b\w+\b', sentence.lower())
        if not sentence_words: continue
        score = sum(word_freq[word] for word in sentence_words)
        sentence_scores[i] = score / len(sentence_words) * (1 - (i / len(sentences)))
    top_sentence_indices = sorted(sentence_scores, key=sentence_scores.get, reverse=True)[:num_sentences]
    top_sentence_indices.sort()
    summary = " ".join([sentences[i] for i in top_sentence_indices])
    return summary



# === MODIFIED FUNCTION 1 START ===
def chunk_texts_from_pdf(pdf_path, para_offset):
    """
    Chunks a PDF into sentences and creates mappings to their paragraph's global index.
    """
    loader = PyMuPDFLoader(pdf_path)
    chunks, metas = [], []
    file_para_list = []
    file_para_idx_map = []
    
    pages = loader.load()
    file_para_counter = 0
    
    for p_i, page in enumerate(pages):
        paragraphs = [p.strip() for p in page.page_content.split("\n\n") if p.strip()]
        for para in paragraphs:
            para_sents = [s.strip() for s in re.split(r"(?<=[.!?])\s+", para) if s.strip()] #Relies on punctuation followed by whitespace. May miss edge cases
            if not para_sents:
                continue
            
            file_para_list.append(para_sents)
            # The absolute paragraph index is the file's paragraph count + the offset from previous files.
            absolute_para_idx = para_offset + file_para_counter
            
            for sent_idx, sent in enumerate(para_sents):
                chunks.append(sent)
                metas.append({"file": os.path.basename(pdf_path), "page": p_i + 1})
                # Map this chunk to its absolute paragraph index and its index within that paragraph.
                file_para_idx_map.append((absolute_para_idx, sent_idx))
            
            file_para_counter += 1
            
    return chunks, metas, file_para_list, file_para_idx_map


# === TFIDF  ===
def build_tfidf_manual(pdf_folder):
    """
    Loads all PDFs, calculates TF-IDF, and builds all necessary data structures
    including the corrected paragraph and sentence mappings.
    """
    file_summaries = {}
    pdf_files = sorted(glob.glob(os.path.join(pdf_folder, "*.pdf")))
    file_texts = []
    
    # First loop: Load all text content for IDF calculation and summaries.
    for pdf_path in pdf_files:
        loader = PyMuPDFLoader(pdf_path)
        file_text = " ".join(page.page_content for page in loader.load())
        file_texts.append(file_text)
        try:
            if_needed_file_summary_can_be_generated  #file_summaries[os.path.basename(pdf_path)] = short_summary(file_text)
        except Exception:
            file_summaries[os.path.basename(pdf_path)] = ""

    # Calculate TF-IDF vocabulary and weights
    file_tokens_per_file = [set(extract_tokens(t)) for t in file_texts]
    vocab = sorted(set().union(*file_tokens_per_file)) # call  each set in the list as a separate argument.
    vocab_idx = {tok: i for i, tok in enumerate(vocab)}
    N = len(file_tokens_per_file)
    df = Counter()
    for toks in file_tokens_per_file:
        for tok in set(toks):
            df[tok] += 1
    idf = {tok: np.log((N + 1) / (1 + df[tok])) + 1 for tok in vocab}

    # Second loop: Chunk texts and create correct mappings for context retrieval.
    all_chunks, all_metas, all_tokens = [], [], []
    all_para_lists, all_para_idx_maps = [], []
    chunk_file_ids = []
    
    para_offset = 0
    for f_id, pdf_path in enumerate(pdf_files):
        # Pass the current paragraph offset to maintain a global index.
        chunks, metas, para_list, para_idx_map = chunk_texts_from_pdf(pdf_path, para_offset)
        
        all_chunks.extend(chunks)
        all_metas.extend(metas)
        all_para_lists.extend(para_list)
        all_para_idx_maps.extend(para_idx_map)
        chunk_file_ids.extend([f_id] * len(chunks))
        
        for ch in chunks:
            all_tokens.append(extract_tokens(ch))
            
        # Increment the offset by the number of paragraphs found in the current file.
        para_offset += len(para_list)

    # Calculate TF-IDF vectors for each chunk
    chunk_vectors = []
    for tokens in all_tokens:
        tf = Counter(tokens)
        v = np.zeros(len(vocab))
        for tok, freq in tf.items():
            if tok in idf:
                v[vocab_idx[tok]] = freq * idf[tok]
        chunk_vectors.append(v)
    
    chunk_vectors = np.array(chunk_vectors, dtype=np.float32)
    norms = np.linalg.norm(chunk_vectors, axis=1, keepdims=True) + 1e-8
    chunk_vectors /= norms
    
    return (vocab, vocab_idx, idf, chunk_vectors, all_chunks, all_metas, all_tokens, chunk_file_ids, file_summaries, pdf_files, all_para_lists, all_para_idx_maps)


def tfidf_vec_for_text(text, vocab, vocab_idx, idf):
    tokens = extract_tokens(text)
    tf = Counter(tokens)
    v = np.zeros(len(vocab))
    for tok, freq in tf.items():
        if tok in idf:
            v[vocab_idx[tok]] = freq * idf[tok]
    v /= (np.linalg.norm(v) + 1e-8)
    return v, tokens

def cosine(u, v):
    denom = np.linalg.norm(u) * np.linalg.norm(v)
    return float(np.dot(u, v) / (denom + 1e-8)) if denom else 0.0

def context_window(para_sents, idx, n1, n2):
    start, end = max(idx-n1, 0), min(idx+n2+1, len(para_sents))
    return " ".join(para_sents[start:end])

def build_dense_store(texts, metas):
    return FAISS.from_texts(texts, embeddings, metadatas=metas , distance_strategy="COSINE")
#
def dense_hits(q, store, top_k):
    docs_and_scores = store.similarity_search_with_score(q, k=top_k)
    # Convert distances explicitly to cosine similarities
    converted_results = [(doc, 1 - (dist / 2)) for doc, dist in docs_and_scores]
    return converted_results  # correctly sorted by FAISS already
#def dense_hits(q, store, top_k):
    #return store.similarity_search_with_score(q, k=top_k)

def sparse_hits(q_vec, chunk_vectors, all_chunks, all_metas, all_tokens, min_ngrams=MIN_NGRAMS, q_tokens=None, top_k=TOP_K_SPARSE):
    sims = chunk_vectors @ q_vec
    idxs = np.argsort(-sims)[:top_k]
    results = []
    for i in idxs:
        overlap = list(set(q_tokens) & set(all_tokens[i]))
        if sims[i] <= 0:
            continue
        if len(overlap) < min_ngrams:
            continue
        manual_cos = cosine(chunk_vectors[i], q_vec)
        results.append((all_chunks[i], float(sims[i]), float(manual_cos), all_metas[i], overlap, i))
    return results

def collect_context_sparse(hits, para_lists, para_idx_maps, n1, n2):
    ctxs = []
    metas = []
    for (text, sim, manual_cos, meta, overlap, i) in hits:
        para_idx, sent_idx = para_idx_maps[i]
        ctx_text = context_window(para_lists[para_idx], sent_idx, n1, n2)
        ctxs.append(ctx_text)
        metas.append(meta)
    return ctxs, metas

def collect_context_dense(hits, para_lists, para_idx_maps, n1, n2):
    ctxs, metas = [], []
    for i, (doc, sim) in enumerate(hits):
        meta = doc.metadata if hasattr(doc, "metadata") else {}
        found = False
        for j, para_sents in enumerate(para_lists):
            for idx, sent in enumerate(para_sents):
                if sent == doc.page_content: # comparson may fail 
                    ctx_text = context_window(para_sents, idx, n1, n2)
                    ctxs.append(ctx_text)
                    metas.append(meta)
                    found = True
                    break
            if found: break
        if not found:
            ctxs.append(doc.page_content)
            metas.append(meta)
    return ctxs, metas

def retrieve_and_collect_dense_context(q, store, para_lists, para_idx_maps, n1, n2, top_k):
    """Retrieves dense hits for a query and collects context windows."""
    pdf_dense_raw = dense_hits(q, store, top_k)
    pdf_dense_ctx, pdf_dense_metas = collect_context_dense(pdf_dense_raw, para_lists, para_idx_maps, n1, n2)
    return pdf_dense_ctx, pdf_dense_metas, pdf_dense_raw

def retrieve_and_collect_sparse_context(q, vocab, vocab_idx, idf, chunk_vectors, all_chunks, all_metas, all_tokens, para_lists, para_idx_maps, n1, n2, top_k):
    """Calculates query vector, retrieves sparse hits, and collects context windows."""
    q_vec, q_tokens = tfidf_vec_for_text(q, vocab, vocab_idx, idf)
    pdf_sparse_raw = sparse_hits(q_vec, chunk_vectors, all_chunks, all_metas, all_tokens, q_tokens=q_tokens, top_k=top_k)
    pdf_sparse_ctx, pdf_sparse_metas = collect_context_sparse(pdf_sparse_raw, para_lists, para_idx_maps, n1, n2)
    return pdf_sparse_ctx, pdf_sparse_metas, pdf_sparse_raw, q_tokens

def ask(
    q,
    dense_store,
    vocab, vocab_idx, idf, chunk_vectors, all_chunks, all_metas, all_tokens, file_summaries,
    chat_history, pdf_files, para_lists, para_idx_maps
):
    global TURN
    TURN += 1
    timestamp = datetime.utcnow().isoformat(timespec="seconds")

    pdf_dense_ctx, pdf_dense_metas, pdf_dense_raw = retrieve_and_collect_dense_context(
        q, dense_store, para_lists, para_idx_maps, PDF_DENSE_N1, PDF_DENSE_N2, TOP_K_DENSE
    )
    pdf_sparse_ctx, pdf_sparse_metas, pdf_sparse_raw, q_tokens = retrieve_and_collect_sparse_context(
        q, vocab, vocab_idx, idf, chunk_vectors, all_chunks, all_metas, all_tokens, 
        para_lists, para_idx_maps, PDF_SPARSE_N1, PDF_SPARSE_N2, TOP_K_SPARSE
    )
    
    # Chat RAG
    chat_chunks, chat_metas, chat_tokens_list, chat_para_lists, chat_para_idx_maps = [], [], [], [], []
    chat_sentence_sources = []
    for turn_idx, entry in enumerate(chat_history[::-1]):
        sents = [s for s in re.split(r"(?<=[.!?])\s+", entry) if s.strip()]
        for idx, s in enumerate(sents):
            if s and s not in chat_sentence_sources:
                chat_chunks.append(s)
                chat_metas.append({"turn": turn_idx, "role": "chat"})
                chat_tokens_list.append(extract_tokens(s))
                chat_para_lists.append(sents)
                chat_para_idx_maps.append((len(chat_para_lists)-1, idx))
                chat_sentence_sources.append(s)
            if len(chat_chunks) >= MAX_CHAT_HISTORY:
                break
        if len(chat_chunks) >= MAX_CHAT_HISTORY:
            break
    chat_dense_raw = []
    if chat_chunks:
        chat_dense_store = build_dense_store(chat_chunks, chat_metas)
        chat_dense_raw = dense_hits(q, chat_dense_store, min(len(chat_chunks), TOP_K_DENSE))
    chat_sparse_raw = []
    if chat_chunks:
        chat_vecs = []
        q_vec, _ = tfidf_vec_for_text(q, vocab, vocab_idx, idf) 
        for ch in chat_chunks:
            tokens = extract_tokens(ch)
            tf = Counter(tokens)
            v = np.zeros(len(vocab))
            for tok, freq in tf.items():
                if tok in idf:
                    v[vocab_idx[tok]] = freq * idf[tok]
            chat_vecs.append(v)
        chat_vecs = np.array(chat_vecs, dtype=np.float32)
        norms = np.linalg.norm(chat_vecs, axis=1, keepdims=True) + 1e-8
        chat_vecs /= norms
        chat_sparse_raw = sparse_hits(q_vec, chat_vecs, chat_chunks, chat_metas, chat_tokens_list, q_tokens=q_tokens, top_k=min(len(chat_chunks), TOP_K_SPARSE))
    chat_dense_ctx, chat_dense_metas = collect_context_dense(chat_dense_raw, chat_para_lists, chat_para_idx_maps, CHAT_DENSE_N1, CHAT_DENSE_N2)
    chat_sparse_ctx, chat_sparse_metas = collect_context_sparse(chat_sparse_raw, chat_para_lists, chat_para_idx_maps, CHAT_SPARSE_N1, CHAT_SPARSE_N2)
    # === Manual tfidf log ===
    pdf_sparse_debug = []
    for (text, sim, manual_cos, meta, overlap, i) in pdf_sparse_raw:
        pdf_sparse_debug.append({
            "text": text,
            "faiss_dot": sim,
            "manual_cosine": manual_cos,
            "overlap": overlap
        })
    chat_sparse_debug = []
    for (text, sim, manual_cos, meta, overlap, i) in chat_sparse_raw:
        chat_sparse_debug.append({
            "text": text,
            "faiss_dot": sim,
            "manual_cosine": manual_cos,
            "overlap": overlap
        })
    manual_log_obj = {
        "turn": TURN,
        "timestamp": timestamp,
        "question": q,
        "question_tokens": q_tokens,
        "pdf_sparse_debug": pdf_sparse_debug,
        "chat_sparse_debug": chat_sparse_debug
    }
    manual_logf = os.path.join(LOG_DIR, f"manual_tfidf_log_turn_{TURN:04d}.json")
    with open(manual_logf, "w", encoding="utf-8") as f:
        json.dump(manual_log_obj, f, ensure_ascii=False, indent=2)
    stories_summary = "\n".join(f"{fname}: {summary}" for fname, summary in file_summaries.items() if summary.strip()) or "<none>"
    context_lines = []
    for m, t in zip(pdf_dense_metas, pdf_dense_ctx):
        context_lines.append(f"{m['file']}: {t}" if 'file' in m else t)
    for m, t in zip(pdf_sparse_metas, pdf_sparse_ctx):
        context_lines.append(f"{m['file']}: {t}" if 'file' in m else t)
    context_block = "\n".join(context_lines) or "<none>"
    chat_hist_lines = []
    for m, t in zip(chat_dense_metas, chat_dense_ctx):
        chat_hist_lines.append(f"{t}")
    for m, t in zip(chat_sparse_metas, chat_sparse_ctx):
        chat_hist_lines.append(f"{t}")
    history_block = "\n".join(chat_hist_lines) or "<none>"
    prompt = PROMPT_TMPL.format(
        turn=TURN,
        context=context_block,
        history=history_block,
        question=q
    )
    answer = llm.invoke(prompt).content.strip() #LLM calls like llm.invoke(prompt) do not have retry logic. Suggestion: Wrap with retries or timeout handling, especially if using APIs with rate limits or flaky connections.
    files_used = list(set([m['file'] for m in pdf_dense_metas + pdf_sparse_metas if 'file' in m]))
    log_obj = {
        "timestamp": timestamp,
        "turn": TURN,
        "question": q,
        "answer": answer,
        "files_used": files_used,
        "params": {
            "pdf_dense_n1": PDF_DENSE_N1,
            "pdf_dense_n2": PDF_DENSE_N2,
            "pdf_sparse_n1": PDF_SPARSE_N1,
            "pdf_sparse_n2": PDF_SPARSE_N2,
            "chat_dense_n1": CHAT_DENSE_N1,
            "chat_dense_n2": CHAT_DENSE_N2,
            "chat_sparse_n1": CHAT_SPARSE_N1,
            "chat_sparse_n2": CHAT_SPARSE_N2,
            "top_k_dense": TOP_K_DENSE,
            "top_k_sparse": TOP_K_SPARSE
        },
        "pdf_dense": {
            "orig_text": [x[0].page_content for x in pdf_dense_raw],
            "orig_sim": [float(x[1]) for x in pdf_dense_raw],
            "ctx_text": pdf_dense_ctx,
            "ctx_meta": pdf_dense_metas
        },
        "pdf_sparse": {
            "orig_text": [x[0] for x in pdf_sparse_raw],
            "orig_sim": [float(x[1]) for x in pdf_sparse_raw],
            "manual_cosine": [float(x[2]) for x in pdf_sparse_raw],
            "ctx_text": pdf_sparse_ctx,
            "ctx_meta": pdf_sparse_metas
        },
        "chat_dense": {
            "orig_text": [x[0].page_content for x in chat_dense_raw],
            "orig_sim": [float(x[1]) for x in chat_dense_raw],
            "ctx_text": chat_dense_ctx,
            "ctx_meta": chat_dense_metas
        },
        "chat_sparse": {
            "orig_text": [x[0] for x in chat_sparse_raw],
            "orig_sim": [float(x[1]) for x in chat_sparse_raw],
            "manual_cosine": [float(x[2]) for x in chat_sparse_raw],
            "ctx_text": chat_sparse_ctx,
            "ctx_meta": chat_sparse_metas
        },
        "llm_prompt": prompt
    }
    with open(os.path.join(LOG_DIR, f"log_turn_{TURN:04d}.json"), "w", encoding="utf-8") as f:
        json.dump(log_obj, f, ensure_ascii=False, indent=2)
    chat_history.append(f"You: {q}\nAI: {answer}")
    return "\n************ Answer ********:\n\n" + answer

# === LOAD DATA AND RUN ===
TURN = 0
CHAT_HISTORY = []
print("Building corpus...")
vocab, vocab_idx, idf, chunk_vectors, all_chunks, all_metas, all_tokens, chunk_file_ids, file_summaries, pdf_files, para_lists, para_idx_maps = build_tfidf_manual(DATA_DIR)
dense_store = build_dense_store(all_chunks, all_metas)
print("Ready. Blank line to send, 'exit' to quit.")
while True:
    print("You: ", end="", flush=True)
    lines = []
    while True:
        line = input()
        if not line.strip(): break
        lines.append(line)
    q = " ".join(lines)
    if q.lower() in {"exit", "quit"}: break
    print(ask(
        q, dense_store, vocab, vocab_idx, idf, chunk_vectors, all_chunks, all_metas, all_tokens,
        file_summaries, CHAT_HISTORY, pdf_files, para_lists, para_idx_maps
    ), "\n")