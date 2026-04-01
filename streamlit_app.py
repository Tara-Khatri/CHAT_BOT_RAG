from __future__ import annotations

import importlib.util
from pathlib import Path

import streamlit as st


def _load_rag_chat_module():
    root = Path(__file__).parent
    path = root / 'assign_16.py'
    spec = importlib.util.spec_from_file_location('rag_capstone', path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f'Could not load module from {path}')
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


@st.cache_resource(show_spinner='Loading web pages, embeddings, and generator...')
def load_pipeline():
    rag = _load_rag_chat_module()
    chunks, sources = rag.load_all_text()
    if not chunks:
        return {'ok': False, 'error': 'No web page text found. Please update URLS in assign_16.py.'}

    embedder = rag.SentenceTransformer(rag.EMBED_MODEL)
    index = rag.build_index(chunks, embedder)

    rag.os.makedirs(rag.MODEL_CACHE, exist_ok=True)
    tokenizer = rag.AutoTokenizer.from_pretrained(
        rag.GEN_MODEL,
        cache_dir=rag.MODEL_CACHE,
        token=rag.HF_TOKEN,
    )

    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token

    model = rag.AutoModelForCausalLM.from_pretrained(
        rag.GEN_MODEL,
        cache_dir=rag.MODEL_CACHE,
        device_map='auto',
        torch_dtype='auto',
        token=rag.HF_TOKEN,
    )

    generator = rag.pipeline('text-generation', model=model, tokenizer=tokenizer)

    return {
        'ok': True,
        'chunks': chunks,
        'sources': sources,
        'embedder': embedder,
        'index': index,
        'generator': generator,
        'rag': rag,
    }


def main():
    st.set_page_config(page_title='RAG Web Chat Bot', page_icon='🤖', layout='centered')
    st.title('RAG Web Chat Bot')
    st.caption('Ask questions about the content of your configured web pages!')

    pipe = load_pipeline()
    if not pipe['ok']:
        st.error(pipe['error'])
        st.stop()

    rag = pipe['rag']
    chunks = pipe['chunks']
    sources = pipe['sources']
    embedder = pipe['embedder']
    index = pipe['index']
    generator = pipe['generator']

    with st.sidebar:
        st.header('Settings')
        k = st.slider('Number of chunks to retrieve (k)', min_value=1, max_value=10, value=3)
        max_tokens = st.slider(
            'Max new tokens',
            min_value=64,
            max_value=1024,
            value=getattr(rag, 'DEFAULT_MAX_NEW_TOKENS', 200),
            step=32,
        )
        st.divider()
        st.markdown(
            "Same pipeline as in 'assign_16.py', but with Streamlit UI and caching for faster load times. "
            'Edit URLS in assign_16.py and reload the page to refresh the chatbot context.'
        )

    if 'messages' not in st.session_state:
        st.session_state.messages = []

    for m in st.session_state.messages:
        if m['role'] == 'user':
            st.markdown(f"**You:** {m['content']}")
        else:
            st.markdown(f"**AI:** {m['content']}")
            if m.get('sources'):
                st.markdown('**Sources used:**')
                for url in m['sources']:
                    st.markdown(f'- [{url}]({url})')

    prompt = st.chat_input('Ask a question about the web pages...')
    if not prompt:
        return

    st.session_state.messages.append({'role': 'user', 'content': prompt})
    with st.chat_message('user'):
        st.markdown(prompt)

    with st.chat_message('AI', avatar='🤖'):
        with st.spinner('Generating response...'):
            hits = rag.retrieval(prompt, embedder, index, chunks, sources, k=k)
            answer = rag.answer_question(prompt, hits, generator, max_new_tokens=max_tokens)
            st.markdown(answer)
            source_urls = sorted(set(h['source'] for h in hits))
            if source_urls:
                st.markdown('**Sources used:**')
                for url in source_urls:
                    st.markdown(f'- [{url}]({url})')
            st.session_state.messages.append({'role': 'assistant', 'content': answer, 'sources': source_urls})


if __name__ == '__main__':
    main()
