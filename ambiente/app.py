import os
import json
from dotenv import load_dotenv
import pdfplumber
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_google_genai import GoogleGenerativeAIEmbeddings, ChatGoogleGenerativeAI
from langchain_community.vectorstores.faiss import FAISS
from langchain.chains.question_answering import load_qa_chain
from langchain.prompts import PromptTemplate
import streamlit as st
import google.generativeai as genai

# Carrega variáveis de ambiente do arquivo .env, que contém informações sensíveis como chaves de API
load_dotenv()
GOOGLE_API_KEY = os.getenv('GOOGLE_API_KEY')  # Obtém a chave da API do Google a partir das variáveis de ambiente

# Configura o cliente da API Google Generative AI usando a chave da API fornecida
genai.configure(api_key=GOOGLE_API_KEY)

# Define o caminho para o arquivo PDF que será processado
pdf_path = "database-site-pecim.pdf"

# Sistema de cache para armazenar respostas anteriores
CACHE_FILE = "response_cache.json"

# Carrega o cache de respostas do arquivo (se existir)
try:
    with open(CACHE_FILE, 'r') as f:
        response_cache = json.load(f)
except FileNotFoundError:
    response_cache = {}

# Função para salvar o cache no arquivo
def save_cache():
    with open(CACHE_FILE, 'w') as f:
        json.dump(response_cache, f)

# Função para ler e extrair todo o texto de um arquivo PDF
import pdfplumber

def extract_text_from_pdf(pdf_file_path):
    extracted_text = ""
    with pdfplumber.open(pdf_file_path) as pdf:
        for page in pdf.pages:
            extracted_text += page.extract_text()
    return extracted_text

# Função para dividir o texto extraído em pedaços menores para processamento
def split_text_into_chunks(text, chunk_size=5000, chunk_overlap=100):
    splitter = RecursiveCharacterTextSplitter(chunk_size=chunk_size, chunk_overlap=chunk_overlap)
    return splitter.split_text(text)

# Função para criar e salvar um banco de vetores (index) para os pedaços de texto
def create_vector_store(chunks):
    embeddings = GoogleGenerativeAIEmbeddings(model="models/text-embedding-004")
    vector_store = FAISS.from_texts(chunks, embedding=embeddings)
    vector_store.save_local("faiss_index")

# Função para configurar o modelo de conversação (chatbot) e o prompt para a cadeia de Pergunta/Resposta (QA)
def setup_qa_chain(language="en"):
    prompt_template = """
    Answer the question as detailed as possible from the provided context, make sure to provide all the details. If the answer is not in
    the provided context, just say, "answer is not available in the context", don't provide the wrong answer.\n\n
    Context:\n {context}\n
    Question: \n{question}\n
    Answer:
    """
    
    if language == "pt":
        prompt_template = """
        Responda à pergunta da forma mais detalhada possível com base no contexto fornecido, garantindo fornecer todos os detalhes. Se a resposta não estiver no contexto fornecido, diga apenas: "a resposta não está disponível no contexto", não forneça a resposta errada.\n\n
        Contexto:\n {context}\n
        Pergunta: \n{question}\n
        Resposta:
        """
    
    model = ChatGoogleGenerativeAI(model="gemini-1.5-flash", client=genai, temperature=0.2)
    prompt = PromptTemplate(template=prompt_template, input_variables=["context", "question"])
    return load_qa_chain(llm=model, chain_type="stuff", prompt=prompt)

# Função para limpar o histórico de mensagens do chat
def clear_chat_history():
    st.session_state.messages = [{"role": "assistant", "content": "O que você deseja saber sobre o PECIM?"}]

# Função para processar a pergunta do usuário e gerar uma resposta com base nos documentos disponíveis
def generate_response(user_question, language="en"):
    if user_question in response_cache:
        return response_cache[user_question]

    embeddings = GoogleGenerativeAIEmbeddings(model="models/text-embedding-004")
    vector_store = FAISS.load_local("faiss_index", embeddings, allow_dangerous_deserialization=True)
    docs = vector_store.similarity_search(user_question)
    chain = setup_qa_chain(language=language)
    context = "\n".join([doc.page_content for doc in docs])

    while True:
        try:
            response = chain({"input_documents": docs, "context": context, "question": user_question}, return_only_outputs=True)
            output_text = response['output_text']

            response_cache[user_question] = output_text
            save_cache()

            return output_text

        except Exception:
            continue

# Função principal para executar o aplicativo Streamlit
def main():
    st.set_page_config(page_title="PECIM's ChatBot", page_icon="🖐", layout="wide")

    if not os.path.exists("faiss_index"):
        with st.spinner("Processando o PDF..."):
            raw_text = extract_text_from_pdf(pdf_path)
            text_chunks = split_text_into_chunks(raw_text)
            create_vector_store(text_chunks)
            st.success("Processamento concluído com sucesso!")

    st.title("Bem-vindo(a) ao PECIM's ChatBot")
    st.markdown("Tire dúvidas e conheça o programa com auxílio do Chatbot do [PECIM](https://www.pecim.unicamp.br/)")

    with st.sidebar:
        st.write("Esta aplicação é não oficial e foi desenvolvida como teste por um estudante do PECIM.")
        st.write("Ela foi desenvolvida para responder apenas questões sobre o PECIM. Qualquer pergunta de outra natureza não será respondida.")
        st.write("As respostas do chatbot são baseadas nas informações contidas no site oficial do programa. Entretanto, as respostas do chat podem conter erros, ausência e falta de precisão durante a interação, principalmente sobre informações mais atualizadas e que não estejam disponíveis no site oficial.")
        st.write("Caso algum erro inesperado aconteça (BUG), considere atualizar a página.")
        st.write("Dúvidas e sugestões: r147725@dac.unicamp.br")
        
        # Adiciona um seletor de idioma
        language = st.selectbox("Selecione o idioma:", ("Português", "Inglês"))
        language_code = "pt" if language == "Português" else "en"

        st.button('Limpar o histórico da conversa', on_click=clear_chat_history)

    if "messages" not in st.session_state:
        clear_chat_history()

    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.markdown(f"**{message['role'].capitalize()}:** {message['content']}")

    if user_prompt := st.chat_input():
        st.session_state.messages.append({"role": "user", "content": user_prompt})
        with st.chat_message("user"):
            st.markdown(f"**Você:** {user_prompt}")

        if st.session_state.messages[-1]["role"] != "assistant":
            with st.chat_message("assistant"):
                with st.spinner("Pensando..."):
                    response = generate_response(user_prompt, language=language_code)
                    st.session_state.messages.append({"role": "assistant", "content": response})
                    st.markdown(f"**Chatbot:** {response}")

if __name__ == "__main__":
    main()
