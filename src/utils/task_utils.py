import configparser
import os

import dotenv
import yaml
from google.cloud import bigquery
from google.oauth2 import service_account

# -----------------------
# 1. Read configuration file
# -----------------------
config = configparser.ConfigParser()
if os.path.exists("config/embedding_config.config"):
    config.read("config/embedding_config.config")
else:
    config.read("src/utils/ml/embeddings/config/embedding_config.config")
# Keep legacy keys for fallback/defaults
table_name = config.get("TABLE", "table_name", fallback=None)
destination_table_name = config.get("TABLE", "destination_table_name", fallback=None)
text_field = config.get("TABLE", "text_field", fallback=None)
model_name_default = config.get("MODEL", "model_name", fallback=None)

dotenv.load_dotenv()
GCP_PROJECT_ID = os.getenv("GCP_PROJECT_ID")
GCP_BIG_QUERY_DATABASE = os.getenv("GCP_BIG_QUERY_DATABASE")
GCP_BIG_QUERY_DATABASE_EMBEDDINGS = os.getenv("GCP_BIG_QUERY_DATABASE_EMBEDDINGS")


# -----------------------
# 2. Connect to BigQuery
# -----------------------
try:
    CREDENTIALS_PATH = os.getenv("CREDENTIALS_PATH_EMBEDDINGS")
    client = bigquery.Client(credentials=service_account.Credentials.from_service_account_file(CREDENTIALS_PATH))
except Exception:
    CREDENTIALS_PATH = os.getenv("GOOGLE_APPLICATION_CREDENTIALS")
    client = bigquery.Client(credentials=service_account.Credentials.from_service_account_file(CREDENTIALS_PATH))


def build_content_expression(content_fields: list[str]) -> str:
    parts = []
    for field in content_fields:
        parts.extend([f"'{field}: '", f"COALESCE(CAST({field} AS STRING), 'null')", "'\\n\\n'"])
    if parts and parts[-1] == "' ":
        parts = parts[:-1]
    return f"CONCAT({', '.join(parts)})"


def build_id_expression(id_fields: list[str]) -> str:
    casted = ", ".join([f"CAST({field} AS STRING)" for field in id_fields])
    return f"ARRAY_TO_STRING([{casted}], '-')"


def build_where_clause(datetime_field: str | None, min_datetime: str | None) -> str:
    if datetime_field and min_datetime:
        return f"WHERE {datetime_field} > '{min_datetime}'"
    return ""


def check_total_length_mb(
    bq_client: bigquery.Client, source_table_fqn: str, content_expr: str, where_clause: str
) -> float:
    query_check = f"""
    SELECT
      SUM(LENGTH({content_expr}))/1000000.0 AS total_length_M,
      SUM(LENGTH({content_expr}))/COUNT(*) AS avg_length,
      APPROX_QUANTILES(LENGTH({content_expr}), 4)[OFFSET(2)] AS median,
      APPROX_QUANTILES(LENGTH({content_expr}), 4)[OFFSET(1)] AS q1,
      APPROX_QUANTILES(LENGTH({content_expr}), 4)[OFFSET(3)] AS q3
    FROM `{source_table_fqn}`
    {where_clause}
    """
    result = bq_client.query(query_check).result()
    row = list(result)[0]
    return row.total_length_M if row.total_length_M is not None else 0


def _get_existing_columns(bq_client: bigquery.Client, table_fqn: str) -> set[str]:
    """Return a set of column names present in the given table."""
    table = bq_client.get_table(table_fqn)
    return {schema_field.name for schema_field in table.schema}


def _build_select_prefix(existing_columns: set[str]) -> str:
    """Return appropriate BigQuery select prefix based on presence of 'content' and 'id' columns.

    - If neither exists: '*'
    - If only one exists: '* EXCEPT(column)'
    - If both exist: '* EXCEPT(content, id)'
    """
    except_cols: list[str] = []
    if "content" in existing_columns:
        except_cols.append("content")
    if "id" in existing_columns:
        except_cols.append("id")
    if not except_cols:
        return "*"
    if len(except_cols) == 1:
        return f"* EXCEPT({except_cols[0]})"
    return "* EXCEPT(content, id)"


def create_embeddings_table_for_task(task: dict):
    source_name = task.get("source_table_name") or task.get("table_name")
    destination_name = task.get("destination_table_name") or task.get("destination_table_name")
    model_for_task = task.get("model_name") or model_name_default
    if not source_name or not destination_name:
        raise ValueError("Both source_table_name and destination_table_name are required in the YAML task.")
    if not model_for_task:
        raise ValueError("model_name not provided in YAML and no default found in INI config.")

    content_fields = task.get("content_fields") or ([] if text_field is None else [text_field])
    if not content_fields:
        raise ValueError("content_fields must be provided in YAML or text_field in INI for fallback.")
    id_fields = task.get("id_fields") or []

    datetime_field = task.get("datetime_field")
    min_datetime = task.get("min_datetime")

    content_expr = build_content_expression(content_fields)
    id_expr = build_id_expression(id_fields) if id_fields else "NULL"
    where_clause = build_where_clause(datetime_field, min_datetime)

    source_table_fqn = f"{GCP_PROJECT_ID}.{GCP_BIG_QUERY_DATABASE}.{source_name}"
    destination_table_fqn = f"{GCP_PROJECT_ID}.{GCP_BIG_QUERY_DATABASE_EMBEDDINGS}.{destination_name}"

    max_characters_millions = config.getint("PARAMS", "max_characters_millions")
    total_length = check_total_length_mb(client, source_table_fqn, content_expr, where_clause)
    print(f"Total length (MB) for {source_name}: {total_length}")
    if total_length > max_characters_millions:
        raise ValueError(
            f"Total characterslength ({total_length} M) exceeds the {max_characters_millions} M characters limit. Table creation skipped."
        )

    existing_columns = _get_existing_columns(client, source_table_fqn)
    select_prefix = _build_select_prefix(existing_columns)

    query_create = f"""
    CREATE OR REPLACE TABLE `{destination_table_fqn}` AS
    SELECT *
    FROM ML.GENERATE_EMBEDDING(
      MODEL `{model_for_task}`,
      (
        SELECT
          {select_prefix},
          {content_expr} AS content,
          {id_expr} AS id
        FROM `{source_table_fqn}`
        {where_clause}
      )
    )
    """

    print(f"Creating table {destination_table_fqn}...")
    print(f"query: {query_create}")
    client.query(query_create).result()
    print("Embeddings table created successfully.")


def insert_embeddings_table_for_task(task: dict):
    source_name = task.get("source_table_name") or task.get("table_name")
    destination_name = task.get("destination_table_name") or task.get("destination_table_name")
    model_for_task = task.get("model_name") or model_name_default
    if not source_name or not destination_name:
        raise ValueError("Both source_table_name and destination_table_name are required in the YAML task.")
    if not model_for_task:
        raise ValueError("model_name not provided in YAML and no default found in INI config.")

    content_fields = task.get("content_fields") or ([] if text_field is None else [text_field])
    if not content_fields:
        raise ValueError("content_fields must be provided in YAML or text_field in INI for fallback.")
    id_fields = task.get("id_fields") or []

    if not id_fields:
        raise ValueError("id_fields must be provided for insert mode to avoid duplicates.")

    datetime_field = task.get("datetime_field")
    min_datetime = task.get("min_datetime")

    content_expr = build_content_expression(content_fields)
    id_expr = build_id_expression(id_fields)
    where_clause = build_where_clause(datetime_field, min_datetime)

    source_table_fqn = f"{GCP_PROJECT_ID}.{GCP_BIG_QUERY_DATABASE}.{source_name}"
    destination_table_fqn = f"{GCP_PROJECT_ID}.{GCP_BIG_QUERY_DATABASE_EMBEDDINGS}.{destination_name}"

    print(f"Checking existing IDs in {destination_table_fqn}...")

    query_existing_ids = f"""
    (SELECT DISTINCT id
    FROM `{destination_table_fqn}`
    WHERE id IS NOT NULL)
    """

    id_exclusion = f"AND {id_expr} NOT IN " + query_existing_ids

    combined_where = where_clause
    if id_exclusion:
        if combined_where:
            combined_where += f" {id_exclusion}"
        else:
            combined_where = f"WHERE {id_exclusion[4:]}"

    total_length = check_total_length_mb(client, source_table_fqn, content_expr, combined_where)
    print(f"Total length (MB) for new rows in {source_name}: {total_length}")

    max_characters_millions = config.getint("PARAMS", "max_characters_millions")
    if total_length > max_characters_millions:
        raise ValueError(
            f"Total characterslength ({total_length} M) exceeds the {max_characters_millions} M characters limit. Insert skipped."
        )

    if total_length == 0:
        print("No new rows to insert.")
        return

    existing_columns = _get_existing_columns(client, source_table_fqn)
    select_prefix = _build_select_prefix(existing_columns)

    query_insert = f"""
    INSERT INTO `{destination_table_fqn}`
    SELECT *
    FROM ML.GENERATE_EMBEDDING(
      MODEL `{model_for_task}`,
      (
        SELECT
          {select_prefix},
          {content_expr} AS content,
          {id_expr} AS id
        FROM `{source_table_fqn}`
        {combined_where}
      )
    )
    """

    print(f"Inserting new rows into {destination_table_fqn}...")
    print(f"query: {query_insert}")
    client.query(query_insert).result()
    print("New embeddings inserted successfully.")


def _derive_destination_suffix_from_model(model_name: str) -> str:
    if not model_name:
        return "embeddings"
    suffix = model_name
    if suffix.startswith("models.text_"):
        suffix = suffix[len("models.text_") :]
    if suffix == "embedding":
        suffix = "embeddings"
    return suffix


def _parse_table_names(value) -> list[str]:
    if value is None:
        return []
    if isinstance(value, list):
        return [str(v).strip() for v in value if str(v).strip()]
    return [part.strip() for part in str(value).split(",") if part.strip()]


def load_tasks_from_yaml(yaml_path: str, tasks_id: str = "tasks_profiles_no_metadata") -> list[dict]:
    with open(yaml_path, "r", encoding="utf-8") as f:
        data = yaml.safe_load(f) or {}
    raw_tasks = data.get(tasks_id)

    if isinstance(raw_tasks, list):
        return raw_tasks

    if isinstance(raw_tasks, dict):
        shared = dict(raw_tasks)
        table_names = _parse_table_names(shared.pop("table_names", None))
        tasks: list[dict] = []
        shared_model = shared.get("model_name") or model_name_default or "models.text_embedding"
        suffix = _derive_destination_suffix_from_model(shared_model)
        table_suffix_name = shared.pop("table_suffix_name", None)
        for table in table_names:
            task = dict(shared)
            task["source_table_name"] = table
            task["destination_table_name"] = f"{table}{table_suffix_name}_{suffix}"
            task.setdefault("model_name", shared_model)
            tasks.append(task)
        return tasks

    return []


__all__ = [
    "build_content_expression",
    "build_id_expression",
    "build_where_clause",
    "check_total_length_mb",
    "create_embeddings_table_for_task",
    "insert_embeddings_table_for_task",
    "load_tasks_from_yaml",
]
