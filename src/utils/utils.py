import logging
import os

import dotenv
from google.cloud import bigquery
from google.oauth2 import service_account

dotenv.load_dotenv()
CREDENTIALS_PATH = os.getenv("CREDENTIALS_PATH_EMBEDDINGS")

# Configurar logger solo para pantalla
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger("embedding_utils")


def bq_query_embeddings(query):
    try:
        from src.utils.bd_utils.big_query_utils import bq_query

        logger.info("Using bq_query from big_query_utils")
        return bq_query(query)
    except:
        logger.info("Using bq_query from embeddings utils")

        credentials = service_account.Credentials.from_service_account_file(CREDENTIALS_PATH)

        # Crear el cliente de BigQuery con las credenciales
        client = bigquery.Client(credentials=credentials, project=credentials.project_id)
        print(f"final query = {query}")
        query_job = client.query(query)
        df = query_job.to_dataframe(create_bqstorage_client=True)
        return df


def load_query_embeddings(file_name):
    """
    Loads the SQL query from a file located in the 'queries' folder.
    """

    try:
        from src.utils.bd_utils.general_bd_utils import load_query

        logger.info("Using load_query from general_bd_utils")
        return load_query(file_name)
    except:
        logger.info("Using load_query from embeddings utils")
        file_path = os.path.join("queries", file_name)
        try:
            with open(file_path, "r", encoding="utf-8") as file:
                query = file.read()
                print(query)
                return query
        except Exception as ex:
            print(f"Error reading query file {file_name}: {ex}")
            return ""
