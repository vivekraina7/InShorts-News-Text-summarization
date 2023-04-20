import streamlit as st
from transformers import TFBigBirdPegasusForConditionalGeneration, BigBirdPegasusTokenizer

st.set_page_config(page_title="Text Summarization App")

st.title("Text Summarization App")

tokenizer = BigBirdPegasusTokenizer.from_pretrained("google/bigbird-pegasus-large-pubmed")
model = TFBigBirdPegasusForConditionalGeneration.from_pretrained("model")

text = st.text_area("Enter the text to be summarized:", height=200)

if st.button("Summarize"):
    inputs = tokenizer.encode_plus(text, return_tensors='tf', padding='max_length', max_length=1024, truncation=True)
    summary_ids = model.generate(inputs['input_ids'], attention_mask=inputs['attention_mask'], max_length=256, min_length=56, num_beams=4, no_repeat_ngram_size=2)
    summary = tokenizer.decode(summary_ids[0], skip_special_tokens=True)
    st.write(summary)
