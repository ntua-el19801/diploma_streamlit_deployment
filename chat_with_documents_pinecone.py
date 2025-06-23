# Install all libraries by running in the terminal: pip install -q -r ./requirements.txt
import streamlit as st
from streamlit_chat import message
# from langchain_openai import OpenAIEmbeddings
from langchain.embeddings import AzureOpenAIEmbeddings
from langchain_community.vectorstores import Chroma

from langchain.schema import(
    SystemMessage,
    HumanMessage,
    AIMessage
)




# loading PDF, DOCX and TXT files as LangChain Documents
def load_document(file):
    import os
    name, extension = os.path.splitext(file)

    if extension == '.pdf':
        # from langchain.document_loaders import PyPDFLoader
        # print(f'Loading {file}')
        # loader = PyPDFLoader(file)
        # from langchain.document_loaders import PyPDFLoader
        from langchain_pymupdf4llm import PyMuPDF4LLMLoader
        from langchain_community.document_loaders.parsers import LLMImageBlobParser
        # from langchain_openai import AzureChatOpenAI
        # import google.generativeai as genai
        from langchain_google_genai import ChatGoogleGenerativeAI
        print(f'Loading {file}')
        loader = PyMuPDF4LLMLoader(
            file,
            mode="page",
            extract_images=True,
            images_parser=LLMImageBlobParser(
                prompt= '''Είσαι ένας βοηθός με αποστολή να συνοψίζεις εικόνες για σκοπούς ανάκτησης. Απαντάς στην ελληνική γλώσσα.
                Σύνοψη: Κατάγραψε με ακρίβεια και συντομία τα δεδομένα της εικόνας ώστε να είναι εύκολη η ανάκτησή τους.
                Περιεχόμενο: Συμπερίλαβε όλα τα εμφανιζόμενα στοιχεία χωρίς σχόλια, αξιολογήσεις ή ερμηνείες.
                Διάταξη: Διατήρησε τη δομή του πίνακα ή των δεδομένων όσο γίνεται.
                Μορφή: Η απάντηση να είναι σε markdown χωρίς να περιέχει επεξηγηματικό κείμενο ή περιγράμματα κώδικα όπως ```.''',
                model=AzureChatOpenAI(
                azure_deployment="gpt-4",  # or your deployment
                api_version="2025-01-01-preview",  # or your api version
                # temperature=0,
                max_tokens=4096,
                # timeout=None,
                # max_retries=2,
                # # other params...
                ),
                # model=ChatGoogleGenerativeAI(model='gemini-1.5-flash', temperature=0.9)
            ),
            table_strategy="lines",
        )
    elif extension == '.docx':
        from langchain.document_loaders import Docx2txtLoader
        print(f'Loading {file}')
        loader = Docx2txtLoader(file)
    elif extension == '.txt':
        from langchain.document_loaders import TextLoader
        loader = TextLoader(file)
    else:
        print('Document format is not supported!')
        return None

    data = loader.load()
    return data

# loading PDF, DOCX and TXT files as LangChain Documents
def load_document_OCR(file):
    import os
    name, extension = os.path.splitext(file)

    if extension == '.pdf':
        from langchain_pymupdf4llm import PyMuPDF4LLMLoader
        from langchain_community.document_loaders.parsers import TesseractBlobParser
        print(f'Loading {file}')
        loader = PyMuPDF4LLMLoader(
            file,
            mode="page",
            extract_images=True,
            images_parser=TesseractBlobParser(),
            table_strategy="lines",
        )
    elif extension == '.docx':
        from langchain.document_loaders import Docx2txtLoader
        print(f'Loading {file}')
        loader = Docx2txtLoader(file)
    elif extension == '.txt':
        from langchain.document_loaders import TextLoader
        loader = TextLoader(file)
    else:
        print('Document format is not supported!')
        return None

    data = loader.load()
    return data


# splitting data in chunks
def chunk_data(data, chunk_size=256, chunk_overlap=20):
    from langchain.text_splitter import RecursiveCharacterTextSplitter
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=chunk_size, chunk_overlap=chunk_overlap)
    chunks = text_splitter.split_documents(data)
    return chunks


# # create embeddings using OpenAIEmbeddings() and save them in a Chroma vector store
# def create_embeddings(chunks):
#     embeddings = OpenAIEmbeddings(model='text-embedding-3-small', dimensions=1536)  # 512 works as well
#     vector_store = Chroma.from_documents(chunks, embeddings)

#     # if you want to use a specific directory for chromadb
#     # vector_store = Chroma.from_documents(chunks, embeddings, persist_directory='./mychroma_db')
#     return vector_store

def insert_or_fetch_embeddings(index_name, chunks=None):
    # importing the necessary libraries and initializing the Pinecone client
    # import pinecone
    # from langchain_community.vectorstores import Pinecone

    # from langchain_openai import OpenAIEmbeddings
    from langchain.embeddings import AzureOpenAIEmbeddings
    from pinecone import Pinecone
    from pinecone import ServerlessSpec
    from langchain_pinecone import PineconeVectorStore

    
    # pc = pinecone.Pinecone()
    pc = Pinecone()




    # embeddings = OpenAIEmbeddings(model='text-embedding-3-small', dimensions=1536)  # 512 works as well
    embeddings = AzureOpenAIEmbeddings(
    # openai_api_key="from .env"
    # azure_endpoint="from .env"
    openai_api_version="2024-02-01",
    azure_deployment="text-embedding-3-large",
    model="text-embedding-3-large",
    chunk_size=2800
    )




    # loading from existing index
    existing_index_names = [index["name"] for index in pc.list_indexes()]
    if index_name in existing_index_names:
        print(f'Index {index_name} already exists. Loading embeddings ... ', end='')
        # vector_store = Pinecone.from_existing_index(index_name, embeddings)
        vector_store = PineconeVectorStore.from_existing_index(index_name=index_name, embedding=embeddings)
        print('Ok')
    else:
        # creating the index and embedding the chunks into the index 
        print(f'Creating index {index_name} and embeddings ...', end='')

        # creating a new index
        pc.create_index(
            name=index_name,
            dimension=3072,
            metric='cosine',
            spec=ServerlessSpec(
                cloud="aws",
                region="us-east-1"
        ) 
        )
        
        # processing the input documents, generating embeddings using the provided `OpenAIEmbeddings` instance,
        # inserting the embeddings into the index and returning a new Pinecone vector store object. 
        # vector_store = Pinecone.from_documents(chunks, embeddings, index_name=index_name)
        vector_store = PineconeVectorStore(index_name=index_name, embedding=embeddings)
        vector_store.add_documents(documents=chunks)
        print('Ok')
        
    return vector_store

def delete_pinecone_index(index_name='all'):
    import pinecone
    pc = pinecone.Pinecone()
    
    if index_name == 'all':
        indexes = pc.list_indexes().names()
        print('Deleting all indexes ... ')
        for index in indexes:
            pc.delete_index(index)
        print('Ok')
    else:
        print(f'Deleting index {index_name} ...', end='')
        pc.delete_index(index_name)
        print('Ok')


def ask_and_get_answer(vector_store, q, k=2):
    from langchain.chains import RetrievalQA
    from langchain_openai import AzureChatOpenAI

    llm = AzureChatOpenAI(
        azure_deployment="gpt-4",
        api_version="2025-01-01-preview",
    )

    retriever = vector_store.as_retriever(search_type='similarity', search_kwargs={'k': k})

    # Retrieve the documents separately
    docs = retriever.get_relevant_documents(q)

    # Optional: print or inspect the retrieved chunks
    print("Retrieved Chunks:")
    for i, doc in enumerate(docs):
        print(f"\nChunk {i + 1}:\n{doc.page_content}")

    # Then build the chain
    chain = RetrievalQA.from_chain_type(llm=llm, chain_type="stuff", retriever=retriever)

    answer = chain.invoke(q)
    return answer['result'] # return both the answer and the chunks. Actually return only the answer and just print the chunks
    


# calculate embedding cost using tiktoken
def calculate_embedding_cost(texts):
    import tiktoken
    enc = tiktoken.encoding_for_model('text-embedding-3-large')
    total_tokens = sum([len(enc.encode(page.page_content)) for page in texts])
    # check prices here: https://openai.com/pricing
    # print(f'Total Tokens: {total_tokens}')
    # print(f'Embedding Cost in USD: {total_tokens / 1000 * 0.00002:.6f}')
    return total_tokens, total_tokens / 1000 * 0.00002


# clear the chat history from streamlit session state
def clear_history():
    if 'history' in st.session_state:
        del st.session_state['history']


if __name__ == "__main__":
    import os

    # loading the OpenAI api key from .env
    # from dotenv import load_dotenv, find_dotenv
    # load_dotenv(find_dotenv(), override=True)

    st.image('img.png')
    st.subheader('Ψηφιακός Βοηθός για τους Χρήστες του ΑΘΗΝΑ 🤖')

    if 'messages' not in st.session_state:
        st.session_state.messages = []
    
    if not any(isinstance(x, SystemMessage) for x in st.session_state.messages):
        st.session_state.messages.append(
                SystemMessage(content="Είσαι ένας ψηφιακός βοηθός που βοηθάει τους χρήστες του ΑΘΗΝΑ να χρησιμοποιούν την εφαρμογή")
                )

    with st.sidebar:
        # streamlit text input widget for the system message (role)
        # system_message = st.text_input(label='System role')
        # streamlit text input widget for the user message
        user_prompt = st.text_input(label='Send a message')

        # if system_message:
        #     if not any(isinstance(x, SystemMessage) for x in st.session_state.messages):
        #         st.session_state.messages.append(
        #             SystemMessage(content=system_message)
        #             )

        # st.write(st.session_state.messages)

        # if the user entered a question
        if user_prompt:
            st.session_state.messages.append(
                HumanMessage(content=user_prompt)
            )

            with st.spinner('Working on your request ...'):
                # creating the ChatGPT response
                # response = chat(st.session_state.messages)
                # if q: # if the user entered a question and hit enter
                # standard_answer = "Answer only based on the text you received as input. Don't search external sources. " \
                #                   "If you can't answer then return `I DONT KNOW`."
                # q = f"{q} {standard_answer}"
                if 'vs' not in st.session_state:
                    try:
                        with st.spinner('Loading embeddings from database...'):
                            index_name = index_name = "bank-app-guide"
                            vector_store = insert_or_fetch_embeddings(index_name)
                            st.session_state.vs = vector_store
                            st.success('Embeddings loaded successfully.')
                    except Exception as e:
                        st.error(f"Failed to load embeddings: {e}")
                if 'vs' in st.session_state:  # if there's the vector store (user uploaded, split and embedded a file)
                    vector_store = st.session_state.vs

                    # Initialize k with 3 if not already defined
                    if 'k' not in st.session_state:
                        st.session_state.k = 2

                    k = st.session_state.k  # Now k is guaranteed to exist

                    st.write(f'k: {k}')
                    
                    response = ask_and_get_answer(vector_store, user_prompt, k)
                # adding the response's content to the session state
                st.session_state.messages.append(AIMessage(content=response))

                if len(st.session_state.messages) >= 1:
                    if not isinstance(st.session_state.messages[0], SystemMessage):
                        st.session_state.messages.insert(0, SystemMessage(content='You are a helpful assistant.'))
                # displaying the messages (chat history)
                # for i, msg in enumerate(st.session_state.messages[1:]):
                #     if i % 2 == 0:
                #         message(msg.content, is_user=True, key=f'{i} + 🙂') # user's question
                #     else:
                #         message(msg.content, is_user=False, key=f'{i} +  🤖') # ChatGPT response

        # st.session_state.messages
        # message('this is chatgpt', is_user=False)
        # message('this is the user', is_user=True)
        # text_input for the OpenAI API key (alternative to python-dotenv and .env)
        azure_api_key = st.text_input('Azure OpenAI API Key:', type='password')
        if azure_api_key:
            os.environ['AZURE_OPENAI_API_KEY'] = azure_api_key

        azure_endpoint = st.text_input('Azure OpenAI Endpoint:', type='password')
        if azure_endpoint:
            os.environ['AZURE_OPENAI_ENDPOINT'] = azure_endpoint

        pinecone_api_key = st.text_input('Pinecone API Key:', type='password')
        if pinecone_api_key:
            os.environ['PINECONE_API_KEY'] = pinecone_api_key
        

        # # file uploader widget
        # uploaded_file = st.file_uploader('Upload a file:', type=['pdf', 'docx', 'txt'])

        # # chunk size number widget
        # chunk_size = st.number_input('Chunk size:', min_value=100, max_value=2048, value=512, on_change=clear_history)

        # # k number input widget
        # k = st.number_input('k', min_value=1, max_value=20, value=3, on_change=clear_history)

        # # add data button widget
        # add_data = st.button('Add Data', on_click=clear_history)

        # if uploaded_file and add_data: # if the user browsed a file
        #     with st.spinner('Reading, chunking and embedding file ...'):

        #         # writing the file from RAM to the current directory on disk
        #         bytes_data = uploaded_file.read()
        #         file_name = os.path.join('./', uploaded_file.name)
        #         with open(file_name, 'wb') as f:
        #             f.write(bytes_data)

        #         data = load_document(file_name)
        #         chunks = chunk_data(data, chunk_size=chunk_size)
        #         st.write(f'Chunk size: {chunk_size}, Chunks: {len(chunks)}')

        #         tokens, embedding_cost = calculate_embedding_cost(chunks)
        #         st.write(f'Embedding cost: ${embedding_cost:.4f}')

        #         # creating the embeddings and returning the Chroma vector store
        #         # vector_store = create_embeddings(chunks)
        #         delete_pinecone_index()
        #         index_name = "bank-app-guide"
        #         vector_store = insert_or_fetch_embeddings(index_name, chunks)

        #         # saving the vector store in the streamlit session state (to be persistent between reruns)
        #         st.session_state.vs = vector_store
        #         st.success('File uploaded, chunked and embedded successfully.')


    # adding a default SystemMessage if the user didn't entered one
    # if len(st.session_state.messages) >= 1:
    #     if not isinstance(st.session_state.messages[0], SystemMessage):
    #         st.session_state.messages.insert(0, SystemMessage(content='You are a helpful assistant.'))


    # user's question text input widget
    # q = st.text_input('Ρώτησε μια ερώτηση σχετική με το ΑΘΗΝΑ:')
    # if q: # if the user entered a question and hit enter
    #     # standard_answer = "Answer only based on the text you received as input. Don't search external sources. " \
    #     #                   "If you can't answer then return `I DONT KNOW`."
    #     # q = f"{q} {standard_answer}"
    #     if 'vs' not in st.session_state:
    #         try:
    #             with st.spinner('Loading embeddings from database...'):
    #                 index_name = index_name = "bank-app-guide"
    #                 vector_store = insert_or_fetch_embeddings(index_name)
    #                 st.session_state.vs = vector_store
    #                 st.success('Embeddings loaded successfully.')
    #         except Exception as e:
    #             st.error(f"Failed to load embeddings: {e}")
    #     if 'vs' in st.session_state:  # if there's the vector store (user uploaded, split and embedded a file)
    #         vector_store = st.session_state.vs

    #         # Initialize k with 3 if not already defined
    #         if 'k' not in st.session_state:
    #             st.session_state.k = 2

    #         k = st.session_state.k  # Now k is guaranteed to exist

    #         st.write(f'k: {k}')

    #         answer = ask_and_get_answer(vector_store, q, k)

    #         # text area widget for the LLM answer
    #         st.text_area('LLM Answer: ', value=answer)

    #         st.divider()

    #         # if there's no chat history in the session state, create it
    #         if 'history' not in st.session_state:
    #             st.session_state.history = ''

    #         # the current question and answer
    #         # the current question and answer
    #         value = f'Q: {q} \nA: {answer}'

    #         st.session_state.history = f'{value} \n {"-" * 100} \n {st.session_state.history}'
    #         h = st.session_state.history

            # text area widget for the chat history

    # displaying the messages (chat history)
    for i, msg in enumerate(st.session_state.messages[1:]):
        if i % 2 == 0:
            message(msg.content, is_user=True, key=f'{i} + 🙂') # user's question
        else:
            message(msg.content, is_user=False, key=f'{i} +  🤖') # ChatGPT response

# run the app: streamlit run ./chat_with_documents_pinecone.py

    