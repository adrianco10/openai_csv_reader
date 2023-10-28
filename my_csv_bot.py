import os
import streamlit as st
from streamlit_chat import message
from langchain.embeddings.openai import OpenAIEmbeddings
# from langchain.chat_models import ChatOpenAI
from langchain.chains import ConversationalRetrievalChain
from langchain.document_loaders.csv_loader import CSVLoader
from langchain.vectorstores import FAISS
import tempfile
from langchain.llms.openai import OpenAI
from langchain.agents.agent_types import AgentType
from langchain.agents import create_csv_agent

os.environ["OPENAI_API_KEY"] = st.secrets["OPENAI_API_KEY"]

api = st.sidebar.text_input(
    label="Paste your OpenAPI key here",
    type='password'
)

llm = OpenAI(temperature=0, openai_api_key=api)

csv_uploaded = st.sidebar.file_uploader("Upload", type='csv')

try:
    if csv_uploaded:
        # tempfile is used here because CSVloader only accepts file_path
        with tempfile.NamedTemporaryFile(delete=False) as tmp_file:
            tmp_file.write(csv_uploaded.getvalue())
            tmp_file_path = tmp_file.name
        # CSVLoader cuts csv into a list of documents where each doc represents one row of the csv
        loader = CSVLoader(file_path=tmp_file_path, encoding='utf-8', csv_args={'delimiter': ','})
        data = loader.load()

        agent = create_csv_agent(
            llm,
            csv_uploaded,
            verbose=False,
            # zero shot reaction means agent will have no memory. It will not act based on previous info.
            agent_type=AgentType.ZERO_SHOT_REACT_DESCRIPTION,
        )

        # Embeddings allow transforming the parts cut by CSVLoader into vectors which allow the measuring of
        # relatedness of text, which then represent an index based on the content of each row of the given file.
        embeddings = OpenAIEmbeddings()
        vectorstore = FAISS.from_documents(data, embeddings)

        # This allows us to provide the user’s question and conversation history to
        # ConversationalRetrievalChain to generate the chatbot’s response. To remember chat history you need a
        # retriever, which is created from a vectorstore, which in turn is created from embeddings
        chain = ConversationalRetrievalChain.from_llm(llm=llm, retriever=vectorstore.as_retriever())

        # def conversation(query):
        #     result = chain({'question': query, "chat_history": st.session_state['history']})
        #     st.session_state['history'].append((query, result['answer']))
        #
        #     return result['answer']

        # first we check if the keys we want are in the session_state and if not, then we add that key and set a
        # value to it
        if 'history' not in st.session_state:
            st.session_state['history'] = []

        if 'generated' not in st.session_state:
            st.session_state['generated'] = [f"AMA about {csv_uploaded.name}"]

        if 'past' not in st.session_state:
            st.session_state['past'] = ['Hello there!']

        res_container = st.container()  # response container
        input_container = st.container()  # user input

        with input_container:
            with st.form(key='my_form', clear_on_submit=True):
                usr_input = st.text_input("Query:", placeholder="Inquire about your csv here", key='input')
                submit_btn = st.form_submit_button(label="Send")

            if submit_btn and usr_input:
                output = agent.run(usr_input)
                # output = conversation(usr_input)
                st.session_state['past'].append(usr_input)  # add user input to past state
                st.session_state['generated'].append(output)  # add chatbot answers to generated state

        if st.session_state['generated']:
            with res_container:
                for msg in range(len(st.session_state['generated'])):
                    message(st.session_state['past'][msg], is_user=True, key=str(msg) + '_user')
                    message(st.session_state['generated'][msg], key=str(msg))

except ImportError as ie:
    st.warning(ie)
