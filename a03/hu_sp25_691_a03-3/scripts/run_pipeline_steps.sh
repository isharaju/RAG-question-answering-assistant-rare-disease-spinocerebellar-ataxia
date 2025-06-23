# turn off chromadb sending stats back
export ANONYMIZED_TELEMETRY=False

BASEDIR=/Users/isha/Desktop/HU_courses/Harrisburg/CISC_691_Next_GEN_AI/projects/a03/hu_sp25_691_a03-3


# ----------------------------------
#  Step 01: Ingest: parse pdf or text files into cleaned text
# ----------------------------------
# python3 $BASEDIR/main.py step01_ingest --input_filename  Tanifuji_2021_Business_Domain.pdf
# python3 $BASEDIR/main.py step01_ingest --input_filename all

# ----------------------------------
#  Step 02: Generate Embeddings from the cleaned text files
# ----------------------------------
# python3 $BASEDIR/main.py step02_generate_embeddings --input_filename Zhang_et_al_2024_LLMs_cleaned.txt
# python3 $BASEDIR/main.py step02_generate_embeddings --input_filename all

# ----------------------------------
#  Step 03: Store the cleaned text and embeddings in a vector db
# ----------------------------------
# python3 $BASEDIR/main.py step03_store_vectors  --input_filename Zhang_et_al_2024_LLMs_cleaned.txt
# python3 $BASEDIR/main.py step03_store_vectors --input_filename  all

# ----------------------------------
#  Step 04: Retrieve chunks of text and similarity scores
# ----------------------------------
# python3 $BASEDIR/main.py step04_retrieve_chunks --query_args "what are some biomarkers of spinocerebellar ataxia"
# python3 $BASEDIR/main.py step04_retrieve_chunks --query_args "Rare Neurological Diseases"
# python3 $BASEDIR/main.py step04_retrieve_chunks --query_args "What is the age distribution of SCA"
# python3 $BASEDIR/main.py step04_retrieve_chunks --query_args ""
# python3 $BASEDIR/main.py step04_retrieve_chunks --query_args ""


# python3 $BASEDIR/main.py step04_retrieve_chunks --query_args "regional development in japan"

# ----------------------------------
#  Step 05: Run LLM Queries with and without RAG
#  	If the parameter "--use_rag" is not provided, RAG is not performed
# ----------------------------------
# QUERY="Discuss trends in regional development in Japan"
QUERY="What are some biomarkers of spinocerebellar ataxia"
# QUERY="What is the age distribution of SCA"
# QUERY="What are the symptoms of SCA"
# QUERY="What is the estimate d prevalence of SCA in South China"
# QUERY="neurological diseases in South China from 2016 to 2022"
# QUERY="20 RNDs in Guangdong Province"

# python3 $BASEDIR/main.py step05_generate_response  --query_args "$QUERY"
python3 $BASEDIR/main.py step05_generate_response  --query_args "$QUERY"  --use_rag
