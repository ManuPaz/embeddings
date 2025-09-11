DECLARE user_query STRING DEFAULT "{user_query}";

WITH query_embedding AS (
  SELECT ml_generate_embedding_result AS embedding
    FROM ML.GENERATE_EMBEDDING(
    MODEL   `models.{model}`,
        (SELECT user_query AS content))
)
SELECT
  t.content,
  t.ticker as title,
  t.year,
  t.quarter,
  ML.DISTANCE(q.embedding, t.ml_generate_embedding_result, 'COSINE') AS distance
FROM `{project_id}.{database}.{table_name}` t
CROSS JOIN query_embedding q
ORDER BY distance ASC
LIMIT {limit_results};

