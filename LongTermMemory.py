# !pip install sentence_transformers
from sentence_transformers import SentenceTransformer
import pandas as pd
import numpy as np
import psycopg2
from pgvector.psycopg2 import register_vector
import flask_cors
from flask import Flask, request, jsonify
import requests
import json
import re
from flask_cors import cross_origin
from bs4 import BeautifulSoup
from dotenv import load_dotenv
import numpy as np
from flask import Flask, request
from flask_cors import CORS, cross_origin
import http
import tensorflow as tf
import io
from threading import Thread
import os
from sshtunnel import SSHTunnelForwarder
import re
import urllib.request
from transformers import AutoModelForCausalLM, AutoTokenizer, TextStreamer, pipeline
from dotenv import load_dotenv
from PyPDF2 import PdfReader
from langchain.text_splitter import CharacterTextSplitter
from langchain.embeddings import OpenAIEmbeddings, HuggingFaceInstructEmbeddings
from langchain.vectorstores import FAISS
from langchain.memory import ConversationBufferMemory
from langchain.chains import ConversationalRetrievalChain
from langchain import HuggingFacePipeline
from langchain import PromptTemplate, LLMChain
from langchain.callbacks.streaming_stdout import StreamingStdOutCallbackHandler
from langchain_community.chat_models import ChatOpenAI
import cv2
import re
from tempfile import NamedTemporaryFile
from collections import OrderedDict
from PyPDF2 import PdfReader
import urllib.request
import io
from keras.models import load_model  # TensorFlow is required for Keras to work
import numpy as np

from bs4 import BeautifulSoup

import requests
# import cross_origin

# import text file

app = Flask(__name__)
flask_cors.CORS(app)

# model_name = "AI-Growth-Lab/PatentSBERTa"
model_name = "all-MiniLM-L6-v2"
# model_name = None

model = SentenceTransformer("AI-Growth-Lab/PatentSBERTa")
if model_name == "all-MiniLM-L6-v2":
    model = SentenceTransformer('all-MiniLM-L6-v2')
# model = SentenceTransformer('all-MiniLM-L6-v2')
booleanQueryUrl = "http://192.168.1.52:1234/"

api_base = "https://api.endpoints.anyscale.com/v1"
token = "esecret_x984s3eisjv86hm6igfder9x5d"

url = f"{api_base}/chat/completions"

# set prompt = mistral_prompt.txt
with open("mistral_prompt.txt", "r") as f:
    prompt = f.read()

# print(prompt)


def get_pdf_text(URL):
    req = urllib.request.Request(URL, headers={'User-Agent': 'Mozilla/5.0'})
    pdf_doc = urllib.request.urlopen(req).read()
    pdf_doc_bytes = io.BytesIO(pdf_doc)

    text = ""
    pdf_reader = PdfReader(pdf_doc_bytes)
    for page in pdf_reader.pages:
        text += page.extract_text()
    return text


def get_text_chunks(text):
    text_splitter = CharacterTextSplitter(
        separator="\n",
        chunk_size=1000,
        chunk_overlap=200,
        length_function=len
    )
    chunks = text_splitter.split_text(text)
    return chunks


def get_vectorstore(text_chunks):
    embeddings = OpenAIEmbeddings()
    # embeddings = HuggingFaceInstructEmbeddings(model_name="hkunlp/instructor-xl")
    vectorstore = FAISS.from_texts(texts=text_chunks, embedding=embeddings)
    return vectorstore


DEFAULT_SYSTEM_PROMPT = """\
    You are a helpful, respectful and honest assistant. Always answer as helpfully as possible, while being safe. Your answers should not include any harmful, unethical, racist, sexist, toxic, dangerous, or illegal content. Please ensure that your responses are socially unbiased and positive in nature.

    If a question does not make any sense, or is not factually coherent, explain why instead of answering something not correct. If you don't know the answer to a question, please don't share false information."""


def get_prompt(instruction, new_system_prompt=DEFAULT_SYSTEM_PROMPT):
    B_INST, E_INST = "[INST]", "[/INST]"
    B_SYS, E_SYS = "<<SYS>>\n", "\n<</SYS>>\n\n"

    SYSTEM_PROMPT = B_SYS + new_system_prompt + E_SYS
    prompt_template = B_INST + SYSTEM_PROMPT + instruction + E_INST
    return prompt_template


def get_conversation_chain(vectorstore):
    llm = ChatOpenAI()
    # llm = HuggingFaceHub(repo_id="Arc53/DocsGPT-7B",
    #             model_kwargs={"temperature": 0.5, "max_length": 5012})
    # tokenizer = AutoTokenizer.from_pretrained('EleutherAI/gpt-neox-20b')
    # model = AutoModelForCausalLM.from_pretrained(
    #     'Arc53/DocsGPT-7B',
    #     trust_remote_code=True
    # )
    # pipe = pipeline("text2text-generation", model=model, tokenizer=tokenizer)
    # llm = HuggingFacePipeline(pipe, model_kwargs={"temperature": 0})
    # system_prompt = "You are an advanced assistant that excels at translation. "
    # instruction = "Convert the following text from English to French:\n\n {text}"
    # template = get_prompt(instruction, system_prompt)
    # print(template)

    # prompt = PromptTemplate(template=template, input_variables=[
    #                         "chat_history", "user_input"])
    # llm_chain = LLMChain(prompt=prompt, llm=llm)

    memory = ConversationBufferMemory(
        memory_key='chat_history', return_messages=True)

    conversation_chain = ConversationalRetrievalChain.from_llm(
        llm=llm,
        retriever=vectorstore.as_retriever(),
        memory=memory
    )
    return conversation_chain


@app.route('/descriptionFetch')
@cross_origin()
def descriptionFetch():
    load_dotenv()
    id = request.args.get('id')
    # Send a request to the patent URL
    response = requests.get(f"https://patents.google.com/patent/{id}/en")
    response.raise_for_status()  # Raise an exception if the request fails

    soup = BeautifulSoup(response.content, "html.parser")  # Parse the HTML
    description = soup.select_one(
        "section[itemprop='description']").text.strip(),

    text_chunks = get_text_chunks(str(description[0]))

    # create vector store
    vectorstore = get_vectorstore(text_chunks)

    # docs = vectorstore.similarity_search(query, k=2)
    # print("\n".join([doc.page_content for doc in docs]))
    # create conversation chain
    conversation = get_conversation_chain(vectorstore)
    # retriever = vectorstore.as_retriever()
    # docs = retriever.get_relevant_documents(query)
    # print("\n\n\n\n\n\n\n\n\ndocs")
    # print("\n".join([doc.page_content for doc in docs]))
    response = conversation(
        {'question': '''

          Summarize the specific features in a few, short bullet points. Be sure to explain in detail how any mechanisms work. make the examples bold font.
         '''})
    # # In a 3-6 sentences, explain what this invention is to me like I am a 12th grader. Be detailed and make sure to mention all relevant IP. Following each sentence in the summary that you construct, you must directly quote all parts of the document that you used to derive this explanation.
    # In a 3-6 sentences, explain what this invention is to me like I am a 12th grader. Be detailed and make sure to mention all relevant IP. Following each sentence in the summary that you construct, you must directly quote all parts of the document that you used to derive this explanation.
    #   On a scale of 1 to 100, how relevant is the patent to the following patent idea. Return only an Integer. : ''' + query})
    chat_history = response['chat_history']

    # for i, message in enumerate(chat_history):
    #     if i % 2 == 0:
    #         st.write(user_template.replace(
    #             "{{MSG}}", message.content), unsafe_allow_html=True)
    #     else:
    #         st.write(bot_template.replace(
    #             "{{MSG}}", message.content), unsafe_allow_html=True)
    # return the response from the chatbot
    return chat_history[-1].content


if __name__ == '__main__':
    app.run(debug=True, host="0.0.0.0", port=int(os.environ.get("PORT", 5000)))
