#!/usr/bin/env python3
"""
Migration script for embeddings from BigQuery to Pinecone.

This script migrates existing embedding tables from BigQuery to Pinecone,
maintaining the data structure and metadata necessary for semantic searches.
"""

import logging
import os
import time
from dataclasses import dataclass
from typing import Any, Dict, List

import pandas as pd
import yaml
from dotenv import load_dotenv
from google.cloud import bigquery
from pinecone import Pinecone, ServerlessSpec

# Configure logging
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)

load_dotenv()


@dataclass
class MigrationConfig:
    """Configuration for migration."""

    project_id: str
    database: str
    pinecone_api_key: str
    pinecone_environment: str
    credentials: str
    batch_size: int = 100
    max_workers: int = 4


from google.oauth2 import service_account


class BigQueryToPineconeMigrator:
    """Migrator for embeddings from BigQuery to Pinecone."""

    def __init__(self, config: MigrationConfig, yaml_config_path: str = None):
        self.config = config
        self.yaml_config = self.load_yaml_config(yaml_config_path)

        credentials = service_account.Credentials.from_service_account_file(config.credentials)
        self.bq_client = bigquery.Client(project=config.project_id, credentials=credentials)
        self.pc_client = Pinecone(api_key=config.pinecone_api_key)

    def load_yaml_config(self, yaml_config_path: str = None) -> Dict[str, Any]:
        """Load configuration from YAML file."""
        if yaml_config_path is None:
            # Search for file in parent directory
            current_dir = os.path.dirname(os.path.abspath(__file__))
            yaml_config_path = os.path.join(current_dir, "..", "..", "..", "pinecone_migrations.yaml")

        try:
            with open(yaml_config_path, "r", encoding="utf-8") as file:
                config = yaml.safe_load(file)
                logger.info(f"YAML configuration loaded from: {yaml_config_path}")
                return config
        except FileNotFoundError:
            logger.warning(f"YAML configuration file not found: {yaml_config_path}")
            return {}
        except Exception as e:
            logger.error(f"Error loading YAML configuration: {e}")
            return {}

    def get_table_config(self, table_name: str) -> Dict[str, Any]:
        """Get specific configuration for a table."""
        if not self.yaml_config or "tables" not in self.yaml_config:
            logger.warning("No table configuration found in YAML")
            return {}

        return self.yaml_config["tables"].get(table_name, {})

    def create_pinecone_index(self, index_name: str, dimension: int = 768) -> None:
        """Create an index in Pinecone."""
        try:
            # Check if index already exists
            if index_name in [index.name for index in self.pc_client.list_indexes()]:
                logger.info(f"Index {index_name} already exists")
                return

            # Create the index
            self.pc_client.create_index(
                name=index_name,
                dimension=dimension,
                metric="cosine",
                spec=ServerlessSpec(cloud="aws", region="us-east-1"),
            )

            logger.info(f"Index {index_name} created successfully")

            # Wait for index to be ready
            while not self.pc_client.describe_index(index_name).status["ready"]:
                logger.info("Waiting for index to be ready...")
                time.sleep(10)

        except Exception as e:
            logger.error(f"Error creating index {index_name}: {e}")
            raise

    def extract_embeddings_batch(
        self, table_name: str, offset: int, limit: int, id_fields: List[str] = None
    ) -> pd.DataFrame:
        """Extract a batch of embeddings from BigQuery."""
        # Build ORDER BY clause using id_fields
        order_by_clause = ""
        if id_fields:
            # Filter available id_fields that exist in the table
            available_id_fields = id_fields  # We'll let BigQuery handle non-existent fields
            if available_id_fields:
                order_by_clause = f"ORDER BY {', '.join(available_id_fields)}"

        query = f"""
        SELECT *
        FROM `{self.config.project_id}.{self.config.database}.{table_name}`
        {order_by_clause}
        LIMIT {limit}
        OFFSET {offset}
        """
        print(f"query ={query }")
        try:
            result = self.bq_client.query(query)
            df = result.to_dataframe()
            logger.info(f"Extracted {len(df)} records from {table_name} (offset: {offset})")
            return df
        except Exception as e:
            logger.error(f"Error extracting data from {table_name}: {e}")
            raise

    def prepare_pinecone_vectors(
        self,
        df: pd.DataFrame,
        table_name: str,
        metadata_fields: List[str] = None,
        id_fields: List[str] = None,
        text_field: str = None,
    ) -> List[Dict]:
        """Prepare vectors for Pinecone using vectorized pandas operations."""
        time_start = time.time()

        # Filter out rows without embeddings
        embedding_mask = df["ml_generate_embedding_result"].notna()
        df_filtered = df[embedding_mask].copy()

        if len(df_filtered) == 0:
            logger.warning("No valid embeddings found in dataframe")
            return []

        available_id_fields = [field for field in id_fields if field in df_filtered.columns]

        if available_id_fields:
            # Create ID by concatenating id_fields
            df_filtered["custom_id"] = df_filtered[available_id_fields].astype(str).agg("_".join, axis=1)
            logger.info(f"Created custom IDs using fields: {available_id_fields}")

        # Prepare metadata columns
        if metadata_fields:
            # Only include specified metadata fields
            metadata_cols = [col for col in metadata_fields if col in df_filtered.columns]
        else:
            # Include all fields except embedding and custom_id
            metadata_cols = [
                col for col in df_filtered.columns if col not in ["ml_generate_embedding_result", "custom_id"]
            ]

        # Convert metadata to proper types for Pinecone
        metadata_df = df_filtered[metadata_cols].copy()

        # Convert all columns to strings, handling NaN values
        for col in metadata_df.columns:
            metadata_df[col] = metadata_df[col].astype(str).replace("nan", None)
        metadata_df["text"] = df_filtered[text_field]
        # Convert metadata to list of dictionaries
        metadata_list = metadata_df.to_dict("records")

        # Create vectors list using zip (fastest approach)
        vectors = [
            {
                "id": custom_id,
                "values": embedding,
                "metadata": {k: v for k, v in metadata.items() if v is not None},
            }
            for custom_id, embedding, metadata in zip(
                df_filtered["custom_id"],
                df_filtered["ml_generate_embedding_result"],
                metadata_list,
            )
        ]

        print(f"Pinecone vectors created: {len(vectors)}, time: {time.time() - time_start:.2f} seconds")
        return vectors

    def upload_vectors_batch(self, index_name: str, vectors: List[Dict]) -> None:
        """Upload a batch of vectors to Pinecone."""
        index = self.pc_client.Index(index_name)
        assert len(vectors) == self.config.batch_size, f"Expected {self.config.batch_size} vectors, got {len(vectors)}"
        index.upsert(vectors=vectors)
        logger.info(f"Uploaded {len(vectors)} vectors to {index_name}")

    def get_table_schema(self, table_name: str) -> Dict[str, Any]:
        """Get the schema of an embeddings table."""
        table_ref = self.bq_client.dataset(self.config.database).table(table_name)
        table = self.bq_client.get_table(table_ref)

        schema = {
            "columns": [field.name for field in table.schema],
            "num_rows": table.num_rows,
            "size_bytes": table.num_bytes,
        }

        logger.info(f"Schema for {table_name}: {schema}")
        return schema

    def migrate_table(self, table_name: str) -> None:
        """Migrate a complete embeddings table using YAML configuration."""
        logger.info(f"Starting migration of {table_name}")

        # Get table configuration from YAML
        table_config = self.get_table_config(table_name)
        if not table_config:
            logger.error(f"No configuration found for table {table_name}")
            return

        # Extract configuration
        index_name = table_config.get("index_name", table_name.replace("_embeddings", "").replace("_", "-"))
        dimension = table_config.get("dimension", 768)
        fields_to_include = table_config.get("fields_to_include", [])
        fields_to_exclude = table_config.get("fields_to_exclude", [])
        metadata_fields = table_config.get("metadata_fields", [])
        id_fields = table_config.get("id_fields", [])
        text_field = table_config.get("text_field", None)

        logger.info(f"Configuration for {table_name}:")
        logger.info(f"  - Index: {index_name}")
        logger.info(f"  - Dimension: {dimension}")
        logger.info(f"  - Fields to include: {fields_to_include}")
        logger.info(f"  - Fields to exclude: {fields_to_exclude}")
        logger.info(f"  - Metadata fields: {metadata_fields}")
        logger.info(f"  - ID fields: {id_fields}")
        logger.info(f"  - Text field: {text_field}")

        # Get table schema
        schema = self.get_table_schema(table_name)

        # Create index in Pinecone
        self.create_pinecone_index(index_name, dimension=dimension)

        # Calculate total number of batches
        total_rows = schema["num_rows"]
        num_batches = (total_rows + self.config.batch_size - 1) // self.config.batch_size

        logger.info(f"Migrating {total_rows} rows in {num_batches} batches")

        # Migrate in batches
        for batch_num in range(num_batches):
            offset = batch_num * self.config.batch_size

            # Extract data batch
            df = self.extract_embeddings_batch(table_name, offset, self.config.batch_size, id_fields)

            if df.empty:
                logger.info(f"No more data in {table_name}")
                break

            # Filter fields according to configuration
            if fields_to_include:
                # Only include specified fields
                available_fields = [col for col in fields_to_include if col in df.columns]
                df = df[available_fields]
                logger.info(f"Included fields: {available_fields}")

            if fields_to_exclude:
                # Exclude specified fields
                df = df.drop(columns=[col for col in fields_to_exclude if col in df.columns])
                logger.info(f"Excluded fields: {fields_to_exclude}")

            # Prepare vectors for Pinecone
            vectors = self.prepare_pinecone_vectors(df, table_name, metadata_fields, id_fields, text_field)

            if vectors:
                # Upload vectors
                self.upload_vectors_batch(index_name, vectors)

            logger.info(f"Progress: {batch_num + 1}/{num_batches} batches completed")

        logger.info(f"Migration of {table_name} completed")


def main():
    """Main function."""
    # Configuration from environment variables
    config = MigrationConfig(
        project_id=os.getenv("GCP_PROJECT_ID"),
        database=os.getenv("GCP_BIG_QUERY_DATABASE_EMBEDDINGS"),
        pinecone_api_key=os.getenv("PINECONE_API_KEY"),
        pinecone_environment=os.getenv("PINECONE_ENVIRONMENT", "eu-west-1"),  # AWS region where Pinecone runs
        batch_size=int(os.getenv("MIGRATION_BATCH_SIZE", 200)),
        max_workers=int(os.getenv("MIGRATION_MAX_WORKERS", "4")),
        credentials=os.getenv("CREDENTIALS_PATH_EMBEDDINGS"),
    )

    # Validate configuration
    if not all([config.project_id, config.database, config.pinecone_api_key]):
        logger.error("Required environment variables are missing")
        return

    # Create migrator and execute migration
    migrator = BigQueryToPineconeMigrator(config)

    # Migrate specific table using YAML configuration
    table_name = "profiles_df_embeddings"  # Change to the table you want to migrate
    migrator.migrate_table(table_name)


if __name__ == "__main__":
    main()
