import os
from typing import Optional

import pandas as pd
from dotenv import load_dotenv

try:
    from src.utils.ml.embeddings.src.utils.utils import bq_query_embeddings
except:
    from utils.utils import bq_query_embeddings

load_dotenv()


class SemanticSearch:
    """
    Clase para realizar búsquedas semánticas usando embeddings de BigQuery ML.
    """

    def __init__(self, project_id: Optional[str] = None, database: Optional[str] = None):
        """
        Inicializa la clase SemanticSearch.

        Args:
            project_id: ID del proyecto GCP. Si es None, se obtiene de variables de entorno.
            database: Nombre de la base de datos. Si es None, se obtiene de variables de entorno.
        """
        self.project_id = project_id or os.getenv("GCP_PROJECT_ID")
        self.database = database or os.getenv("GCP_BIG_QUERY_DATABASE_EMBEDDINGS")

        if not self.project_id or not self.database:
            raise ValueError("project_id y database son requeridos. Configúralos en variables de entorno o parámetros.")

    def search(
        self,
        query: str,
        user_query: str,
        embedding_type: str = "",
        model: str = "text_embedding",
        limit_results: int = 10,
        **kwargs,
    ) -> pd.DataFrame:
        """
        Realiza una búsqueda semántica usando el query SQL proporcionado.

        Args:
            query: Query SQL con placeholders para formatear
            user_query: Texto de búsqueda del usuario
            embedding_type: Tipo de embedding a usar
            model: Modelo de ML a usar para generar embeddings
            limit_results: Número máximo de resultados
            **kwargs: Parámetros adicionales para el query

        Returns:
            DataFrame con los resultados ordenados por distancia (más similares primero)
        """
        formatted_query = query.format(
            project_id=self.project_id,
            database=self.database,
            embedding_type=embedding_type,
            model=model,
            limit_results=limit_results,
            user_query=user_query,
            **kwargs,
        )

        df = bq_query_embeddings(formatted_query)

        if df is not None and not df.empty:
            df = df.sort_values(by="distance", ascending=True)

        return df
