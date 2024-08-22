import os
import json
from PyPDF2 import PdfReader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_google_genai import GoogleGenerativeAIEmbeddings, ChatGoogleGenerativeAI
from langchain_community.vectorstores import FAISS
from langchain.chains.question_answering import load_qa_chain
from langchain.prompts import PromptTemplate
import streamlit as st
import google.generativeai as genai

# Configura o cliente da API Google Generative AI usando a chave da API fornecida
genai.configure(api_key="AIzaSyABWrsF6-pUZa38AqYmKPUQS1IMm8_LLgM")

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
def extract_text_from_pdf(pdf_path):
    pdf_reader = PdfReader(pdf_path)  # Inicializa o leitor de PDF com o arquivo especificado
    # Extrai o texto de cada página do PDF e o combina em uma única string
    return "".join([page.extract_text() for page in pdf_reader.pages])

# Função para dividir o texto extraído em pedaços menores para processamento
def split_text_into_chunks(text, chunk_size=10000, chunk_overlap=1000):
    # Inicializa o divisor de texto com o tamanho do pedaço e a sobreposição especificados
    splitter = RecursiveCharacterTextSplitter(chunk_size=chunk_size, chunk_overlap=chunk_overlap)
    return splitter.split_text(text)  # Retorna uma lista de pedaços de texto

# Função para criar e salvar um banco de vetores (index) para os pedaços de texto
def create_vector_store(chunks):
    # Configura o modelo de embeddings do Google Generative AI para converter texto em vetores numéricos
    embeddings = GoogleGenerativeAIEmbeddings(model="models/embedding-001")
    # Cria o banco de vetores usando os pedaços de texto e os embeddings gerados
    vector_store = FAISS.from_texts(chunks, embedding=embeddings)
    vector_store.save_local("faiss_index")  # Salva o banco de vetores localmente para uso futuro

# Função para configurar o modelo de conversação (chatbot) e o prompt para a cadeia de Pergunta/Resposta (QA)
def setup_qa_chain():
    # Template do prompt que será usado para gerar as respostas, garantindo respostas detalhadas e corretas
    prompt_template = """
    Answer the question as detailed as possible from the provided context, make sure to provide all the details. If the answer is not in
    the provided context, just say, "answer is not available in the context", don't provide the wrong answer.\n\n
    Context:\n {context}\n
    Question: \n{question}\n
    Answer:
    """
    # Inicializa o modelo de conversação da Google Generative AI com o ajuste de temperatura para controlar a criatividade das respostas
    model = ChatGoogleGenerativeAI(model="gemini-pro", client=genai, temperature=0.1)
    # Cria o prompt de entrada usando o template e especificando as variáveis de entrada (contexto e pergunta)
    prompt = PromptTemplate(template=prompt_template, input_variables=["context", "question"])
    # Carrega a cadeia de QA configurada com o modelo de conversação e o prompt
    return load_qa_chain(llm=model, chain_type="stuff", prompt=prompt)

# Função para limpar o histórico de mensagens do chat
def clear_chat_history():
    # Inicializa o histórico de mensagens com uma mensagem padrão do assistente
    st.session_state.messages = [{"role": "assistant", "content": "O que você deseja saber sobre o PECIM?"}]

# Função para processar a pergunta do usuário e gerar uma resposta com base nos documentos disponíveis
def generate_response(user_question):
    # Verifica se a pergunta já está no cache
    if user_question in response_cache:
        return response_cache[user_question]

    # Carrega os embeddings do modelo de Google Generative AI
    embeddings = GoogleGenerativeAIEmbeddings(model="models/embedding-001")
    # Carrega o banco de vetores salvo anteriormente e permite a desserialização segura
    vector_store = FAISS.load_local("faiss_index", embeddings, allow_dangerous_deserialization=True)
    # Realiza uma busca de similaridade no banco de vetores para encontrar documentos relacionados à pergunta do usuário
    docs = vector_store.similarity_search(user_question)
    # Configura a cadeia de QA (Pergunta/Resposta)
    chain = setup_qa_chain()
    # Combina o conteúdo das páginas dos documentos encontrados para formar o contexto da resposta
    context = "\n".join([doc.page_content for doc in docs])

    # Tentativa de gerar a resposta, repetindo até que o erro não ocorra mais
    while True:
        try:
            # Gera a resposta usando a cadeia de QA com os documentos relevantes e o contexto formado
            response = chain({"input_documents": docs, "context": context, "question": user_question}, return_only_outputs=True)
            output_text = response['output_text']

            # Armazena a resposta no cache
            response_cache[user_question] = output_text
            save_cache()

            return output_text  # Retorna apenas o texto da resposta gerada

        except Exception:
            continue  # Continua a tentativa até que a resposta seja gerada corretamente

# Função principal para executar o aplicativo Streamlit
def main():
    st.set_page_config(page_title="PECIM's ChatBot", page_icon="🖐", layout="wide")  # Configura a página do aplicativo

    # Verifica se o banco de vetores já foi criado e processa o PDF apenas se necessário
    if not os.path.exists("faiss_index"):
        with st.spinner("Processando o PDF..."):  # Exibe um spinner de carregamento enquanto o PDF é processado
            raw_text = extract_text_from_pdf(pdf_path)  # Extrai o texto do PDF
            text_chunks = split_text_into_chunks(raw_text)  # Divide o texto em pedaços menores
            create_vector_store(text_chunks)  # Cria e salva o banco de vetores
            st.success("Processamento concluído com sucesso!")  # Exibe uma mensagem de sucesso após o processamento

    st.title("Bem-vindo(a) ao PECIM's ChatBot")  # Define o título da página
    st.markdown("Tire dúvidas e conheça o programa com auxílio do Chatbot do [PECIM](https://www.pecim.unicamp.br/)")  # Adiciona um texto descritivo

    # Barra lateral com opção para limpar o histórico de conversas
    with st.sidebar:
        st.write("Esta aplicação é não oficial e foi desenvolvida como teste por um estudante do PECIM.")
        st.write("Ela foi desenvolvida para responder apenas questões sobre o PECIM. Qualquer pergunta de outra natureza não será respondida.")
        st.write("As respostas do chatbot são baseadas nas informações contidas no site oficial do programa. Entretando, as respostas do chat podem conter erros, ausência e falta de precisão durante a interação, principalmente sobre informações mais atualizadas e que não estejam disponíveis no site oficial.")
        st.write("Caso algum erro inesperado aconteça (BUG), considere atualizar a página.")
        st.write("Dúvidas e sugestões: r147725@dac.unicamp.br")
        st.button('Limpar o histórico da conversa', on_click=clear_chat_history)

    # Inicializa o histórico de mensagens se ainda não existir na sessão
    if "messages" not in st.session_state:
        clear_chat_history()

    # Exibe o histórico de mensagens no chat
    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.markdown(f"**{message['role'].capitalize()}:** {message['content']}")

    # Captura a entrada do usuário e processa a resposta
    if user_prompt := st.chat_input():
        st.session_state.messages.append({"role": "user", "content": user_prompt})  # Adiciona a pergunta do usuário ao histórico
        with st.chat_message("user"):
            st.markdown(f"**Você:** {user_prompt}")  # Exibe a pergunta do usuário no chat

        if st.session_state.messages[-1]["role"] != "assistant":  # Verifica se a última mensagem é do assistente
            with st.chat_message("assistant"):
                with st.spinner("Pensando..."):  # Exibe um spinner enquanto a resposta é gerada
                    response = generate_response(user_prompt)  # Gera a resposta com base na pergunta do usuário
                    st.session_state.messages.append({"role": "assistant", "content": response})  # Adiciona a resposta ao histórico
                    st.markdown(f"**Chatbot:** {response}")  # Exibe a resposta no chat

if __name__ == "__main__":
    main()
