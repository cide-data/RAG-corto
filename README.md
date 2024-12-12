# RAG-corto
Este repositorio contiene un código diseñado para implementar un sistema de RAG (Retrieval-Augmented Generation) utilizando herramientas de la biblioteca LangChain.  El sistema combina capacidades de recuperación de información con generación de texto para responder a consultas, apoyándose en un documento específico como base de conocimiento.

    import subprocess

#Configurar el entorno e instalar paquetes

    try:
        subprocess.run(['pip', 'install', 'langchain', 'langchain_community', 'langchain-openai',
                        'scikit-learn', 'langchain-ollama', 'pymupdf', 'langchain_huggingface',
                        'faiss-gpu'], check=True)
        print("Paquetes instalados correctamente.")
    except subprocess.CalledProcessError as e:
        print("Error al instalar paquetes:", e)
    
    from langchain.chains import RetrievalQA
    
    from langchain_community.document_loaders import  PyMuPDFLoader
    
    from langchain_community.llms import Ollama
    
    from langchain_community.vectorstores import FAISS
    
    from langchain_huggingface import HuggingFaceEmbeddings
    
    from langchain_text_splitters import CharacterTextSplitter
    
    loader =  PyMuPDFLoader('/content/drive/MyDrive/Colab Notebooks/CIDE/Introducción a la Estadística.pdf')
    
    docs = loader.load()

#Instala y prepara el modelo localmente.

    !curl -fsSL https://ollama.com/install.sh | sh
    
    !pip install colab-xterm
    
    %load_ext colabxterm
    
    %xterm
    
    ##ollama serve
    
    !ollama pull llama3 

#Se divide el contenido del documento en fragmentos manejables utilizando un separador y configurando el tamaño de los fragmentos

    text_splitter = CharacterTextSplitter(
        separator="\n",
        chunk_size=2000,
        chunk_overlap=200
    )
    
    texts = text_splitter.split_documents(docs)

#Se generan vectores a partir de los fragmentos de texto utilizando

    embeddings = HuggingFaceEmbeddings()

#Se crea una base de datos vectorial utilizando FAISS para realizar búsquedas eficientes.

    db = FAISS.from_documents(texts, embeddings)

#Se implementa una cadena de preguntas y respuestas utilizando el modelo

    llm = Ollama(model="llama3")
    
    chain = RetrievalQA.from_chain_type(
        llm,
        retriever=db.as_retriever()
    )

#Se formula una consulta al sistema y se imprime la respuesta generada
    
    question = "que es un estadistico?"
    
    result = chain.invoke({"query": question})
    
    print(result['result'])
