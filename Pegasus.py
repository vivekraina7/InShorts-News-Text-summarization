import streamlit as st
from transformers import AutoTokenizer,AutoModelForSeq2SeqLM #PegasusTokenizer, TFPegasusForConditionalGeneration
import tensorflow as tf

# initialize the Pegasus tokenizer and model
tokenizer = AutoTokenizer.from_pretrained("google/pegasus-cnn_dailymail")
#tokenizer = PegasusTokenizer.from_pretrained('google/pegasus-cnn_dailymail')
model = AutoModelForSeq2SeqLM.from_pretrained("google/pegasus-cnn_dailymail")
#model = TFPegasusForConditionalGeneration.from_pretrained('google/pegasus-cnn_dailymail')


# define a function to preprocess the user input
def preprocess(text):
    inputs = tokenizer(text, truncation=True, padding='longest', max_length=512, return_tensors='tf')
    return inputs

# define a function to generate the summary
def generate_summary(text):
    inputs = preprocess(text)
    summary_ids = model.generate(inputs['input_ids'], num_beams=4, max_length=64, early_stopping=True)
    summary = tokenizer.decode(summary_ids[0], skip_special_tokens=True)
    return summary

# define the Streamlit app
def main():
    # set the app title and description
    st.title('Text Summarization')
    st.markdown('Enter some text and click the button to generate a summary.')

    # create a text input box for the user to enter their text
    text = st.text_input('Enter some text')

    # create a button to generate the summary
    if st.button('Generate Summary'):
        # generate the summary
        summary = generate_summary(text)

        # display the summary
        st.subheader('Summary')
        st.write(summary)

if __name__ == '__main__':
    main()
