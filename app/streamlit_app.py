"""Streamlit UI for the /ask API: original Qwen3.5-4B (Ollama) vs our NF4 + double-quantised model.

1. uv run model-quantization serve     # API on http://127.0.0.1:8000 (loads the NF4-dq model)
2. uv run model-quantization ui        # this app on http://localhost:8501
"""

import os

import pandas as pd
import requests
import streamlit as st

from model_quantization import config

_host = "127.0.0.1" if config.HOST in ("0.0.0.0", "::") else config.HOST
DEFAULT_API = os.getenv("API_URL", f"http://{_host}:{config.PORT}")
LABELS = {"ollama": f"Original — Ollama {config.OLLAMA_MODEL}", "nf4-dq": "NF4 + double quantisation"}

st.set_page_config(page_title="Qwen3.5-4B: original vs NF4-dq", layout="wide")
st.title("Qwen3.5-4B — original vs NF4 + double quantisation")

# ---- sidebar: connection + settings ------------------------------------------------------------
with st.sidebar:
    api = st.text_input("API URL", DEFAULT_API).rstrip("/")
    try:
        h = requests.get(f"{api}/health", timeout=5).json()
        st.success(f"API: {h['status']}")
        st.caption(f"NF4-dq: `{h.get('model_dir')}`")
        o = h.get("ollama", {})
        (st.caption if o.get("reachable") else st.warning)(
            f"Ollama `{o.get('model')}`: {'reachable' if o.get('reachable') else 'NOT reachable — run `ollama serve`'}"
        )
    except requests.RequestException:
        st.error("API not reachable. Start it with `uv run model-quantization serve`.")

    st.divider()
    choice = st.radio("Model", ["both", "ollama", "nf4-dq"],
                      format_func=lambda m: "Both (compare)" if m == "both" else LABELS[m])
    max_new_tokens = st.slider("Max new tokens", 16, 1024, config.MAX_NEW_TOKENS, step=16)
    temperature = st.slider("Temperature", 0.0, 2.0, 0.0, step=0.1)
    thinking = st.checkbox("Thinking mode (much slower on CPU)", value=False)

# ---- ask ---------------------------------------------------------------------------------------
prompt = st.text_area("Your question", "What is model quantisation? Answer in 3 short sentences.", height=110)

if st.button("Ask", type="primary", disabled=not prompt.strip()):
    with st.spinner("Generating… (the NF4 model runs on CPU and can take a few minutes)"):
        try:
            r = requests.post(f"{api}/ask", timeout=None, json={
                "prompt": prompt, "model": choice, "max_new_tokens": max_new_tokens,
                "temperature": temperature, "thinking": thinking,
            })
            r.raise_for_status()
            st.session_state["last"] = r.json()
        except requests.RequestException as e:
            st.error(f"Request failed: {e}")

if last := st.session_state.get("last"):
    st.caption(f"request_id `{last['request_id']}`")
    cols = st.columns(len(last["results"]))
    for col, res in zip(cols, last["results"]):
        with col:
            st.subheader(LABELS.get(res["model"], res["model"]))
            a, b, c = st.columns(3)
            a.metric("Duration", f"{res['duration_sec']:.2f} s", f"{res['duration_min']:.3f} min", delta_color="off")
            b.metric("New tokens", res["new_tokens"])
            c.metric("Tokens / s", f"{res['tokens_per_second']:.2f}")
            st.caption(f"{res['start_time']} → {res['end_time']}")
            if res["status"] == "ok":
                st.markdown(res["response"] or "_(empty answer)_")
            else:
                st.error(res["error"])
    ok = {r["model"]: r for r in last["results"] if r["status"] == "ok"}
    if len(ok) == 2 and ok["ollama"]["duration_sec"]:
        ratio = ok["nf4-dq"]["duration_sec"] / ok["ollama"]["duration_sec"]
        st.info(f"NF4-dq took **{ratio:.1f}×** the time of Ollama for this question.")

# ---- CSV log -----------------------------------------------------------------------------------
st.divider()
st.header("Request log")
try:
    data = requests.get(f"{api}/logs", params={"limit": 1000}, timeout=10).json()
    df = pd.DataFrame(data["rows"])
except (requests.RequestException, ValueError, KeyError):
    df = pd.DataFrame()

if df.empty:
    st.caption("No requests logged yet.")
else:
    for c in ("duration_sec", "duration_min", "tokens_per_second", "new_tokens", "prompt_tokens"):
        df[c] = pd.to_numeric(df[c], errors="coerce")
    st.caption(f"`{data['path']}` — {len(df)} rows, newest first")

    summary = (df[df["status"] == "ok"].groupby("model")
               .agg(calls=("request_id", "count"), avg_sec=("duration_sec", "mean"),
                    avg_min=("duration_min", "mean"), avg_tok_s=("tokens_per_second", "mean"))
               .round(3))
    st.dataframe(summary, width="stretch")

    st.dataframe(
        df[["start_time", "end_time", "duration_sec", "duration_min", "model", "prompt", "response",
            "new_tokens", "tokens_per_second", "status", "request_id"]],
        width="stretch", hide_index=True,
    )
    st.download_button("Download CSV", df.to_csv(index=False).encode("utf-8-sig"),
                       file_name="requests.csv", mime="text/csv")
