DECLARE user_query STRING DEFAULT "{user_query}";

WITH query_embedding AS (
  SELECT ml_generate_embedding_result AS embedding
  FROM ML.GENERATE_EMBEDDING(
    MODEL   `models.{model}`,
    (SELECT user_query AS content)
  )
),
distances AS (
  SELECT
    t.content AS content,
    t.industry,
    t.sector,
    t.country,
    t.companyname AS title,
    ML.DISTANCE(q.embedding, t.ml_generate_embedding_result, 'COSINE') AS distance,
    ROW_NUMBER() OVER (PARTITION BY t.companyname ORDER BY ML.DISTANCE(q.embedding, t.ml_generate_embedding_result, 'COSINE')) AS rn
  FROM `{project_id}.{database}.{table_name}` t
  CROSS JOIN query_embedding q
  WHERE t.description IS NOT NULL
  AND isEtf=False
  AND isFund=False
  AND isAdr=False
  AND isActivelyTrading=True
)
SELECT
  content,
  industry,
  sector,
  country,
  title,
  distance
FROM distances
WHERE rn = 1
ORDER BY distance ASC
LIMIT {limit_results};