# Embeddings Module

Module for creating and managing text embeddings using Google Cloud's Vertex AI models.

## 🎯 Functionality

This module allows you to:
- **Create remote models** in Vertex AI using Google embeddings
- **Generate embeddings** from tables with text fields
- **Compare embedding models** using a Streamlit app
- **Deploy infrastructure** using Terraform

## 🏗️ Structure

```
embeddings/
├── src/                          # Main code
│   ├── embeddings.py             # SemanticSearch class
│   ├── scripts/                  # Utility scripts
│   │   ├── create_embeddings.py  # Create BigQuery embeddings tables
│   │   └── validator.py          # Streamlit app to compare models
│   └── utils/                    # Helper functions
├── terraform/                    # Infrastructure code
├── pyproject.toml                # Dependencies
└── README.md                     # This file
```

## 🚀 Step-by-Step Process

### Step 1: Create Remote Models with Terraform

Terraform creates remote models in Vertex AI that can be called from BigQuery, based on existing Vertex models:

- **text-embedding-005**: Standard embedding model
- **gemini-embedding-001**: Advanced Gemini embedding model

**Costs per 1000 characters:**
- **text-embedding-005**: $0.00002 (0.002 cents)
- **gemini-embedding-001**: $0.00012 (0.012 cents)

### Step 2: Create Embeddings Tables

Once the models are created, use the `create_embeddings.py` script to:

1. **Check character limits** before processing (default: 300M characters)
2. **Calculate costs**:
   - text-embedding-005: 300,000,000 × $0.00002/1000 = $6
   - gemini-embedding-001: 300,000,000 × $0.00012/1000 = $36
3. **Create new BigQuery table** with embeddings for the selected text field

### Step 3: Example Queries

The module includes example queries that work on embedding tables you've created. **Replace the table names** in these queries with your actual embedding tables.

### Step 4: Model Comparison with Streamlit

The `validator.py` script runs a simple Streamlit app that allows you to:

- **Select a query** (on an embedding table)
- **Input search terms** to see results
- **Select two models** to compare embeddings
- **Compare results** from both models side by side

**Note**: You must have created the embedding table with both models beforehand.

## 💰 Costs

Embedding model prices in Vertex AI:

- **text-embedding-005**: $0.00002 per 1000 characters
- **gemini-embedding-001**: $0.00012 per 1000 characters

For updated pricing, check the [official Google Cloud documentation](https://cloud.google.com/vertex-ai/generative-ai/pricing?hl=es-419).

## 📊 Benchmarking

To compare the performance of different embedding models, check the [MTEB leaderboard on Hugging Face](https://huggingface.co/spaces/mteb/leaderboard?utm_source=chatgpt.com).

## 🔧 Configuration

### Required environment variables

```bash
GOOGLE_APPLICATION_CREDENTIALS=path/to/credentials.json
GCP_PROJECT_ID=your-project-id
GCP_BIG_QUERY_DATABASE=your-database
```

### Deploy with Terraform

```bash
cd terraform
terraform init
terraform apply
```

### Create embeddings table

```bash
cd src/utils/ml/embeddings/src/scripts
python create_embeddings.py
```

### Run validation app

```bash
cd src/utils/ml/embeddings/src/scripts
python validator.py
```

## 📝 Notes

- Models are deployed in Vertex AI for scalability
- Embedding tables are created automatically in BigQuery
- The Streamlit app allows validation and result comparison
- Support for multiple Google embedding models
- Character limits and cost calculations are built-in

---

---

## Validator Frontend Walkthrough

The `validator.py` Streamlit app lets you compare two embedding model versions side by side. Each version is typically stored in its own table. You can:

- Compare results from two models with the same user query
- Parameterize the SQL with placeholders like `{user_query}`, `{model}`, `{table_name}`, `{limit_results}`, and optional `{min_date}`
- Inspect distances, overlap, and detailed content/metadata

### Steps





1. Select left and right models and their table names

   Write your text (search query) and set the number of results

![Validator - Model Selection](assets/images/validator_frontend.PNG)

2. Select a predefined query or write a query 

![Validator - Model Selection](assets/images/validator_frontend2.PNG)

3. Run the comparison to view side-by-side tables with results and distance metrics

![Validator - Results Side by Side](assets/images/validator_frontend3.PNG)
![Validator - Metrics](assets/images/validator_frontend4.PNG)

---

## 🚀 Migration to Pinecone & RAG Implementation

### Overview

This module includes a migration system to transfer embeddings from BigQuery to Pinecone, enabling high-performance vector search and RAG (Retrieval-Augmented Generation) applications.

### Migration Process

#### Step 1: Configure Migration

Create a `pinecone_migrations.yaml` file to configure your migration:

```yaml
tables:
  profiles_df_embeddings:
    index_name: "profiles-df"
    dimension: 768
    fields_to_include: []
    fields_to_exclude: []
    metadata_fields: ["company_name", "sector", "description"]
    id_fields: ["company_id", "symbol"]
    text_field: "description"
```

#### Step 2: Run Migration

```bash
cd src/utils/ml/embeddings/src/scripts/migrations
python migrate_to_pinecone.py
```

#### Step 3: Environment Variables

```bash
# Required for migration
GCP_PROJECT_ID=your-project-id
GCP_BIG_QUERY_DATABASE_EMBEDDINGS=your-database
PINECONE_API_KEY=your-pinecone-key
PINECONE_ENVIRONMENT=eu-west-1
CREDENTIALS_PATH_EMBEDDINGS=path/to/credentials.json

# Optional
MIGRATION_BATCH_SIZE=200
MIGRATION_MAX_WORKERS=4
```

### RAG Implementation with Pinecone

#### Quick Start RAG System

The module includes an example of RAG system that integrates VertexAI embeddings with Pinecone:

```bash
cd src/utils/ml/embeddings/src/scripts/examples
python rag_example.py
```

#### RAG System Features

**🔧 Components:**
- **VertexAI Embeddings**: Uses `text-embedding-005` model
- **Pinecone Vector Store**: High-performance vector search
- **Google Generative AI**: `gemini-2.0-flash-lite` for text generation
- **LangChain Integration**: Complete RAG pipeline

#### Example Usage

```python
from src.utils.ml.embeddings.src.scripts.examples.rag_example import RAGSystem

# Initialize RAG system
rag_system = RAGSystem()

# Ask questions
result = rag_system.ask_question("What are the best tech companies in logistics?")
print(result['answer'])

# Perform similarity search
results = rag_system.similarity_search("AI companies", k=5)
for doc in results:
    print(doc.page_content[:200])
```

#### Interactive Dialog Interface

The RAG system includes an interactive command-line interface:

```
🤖 RAG System with VertexAI Embeddings and Pinecone
============================================================

🚀 Initializing RAG System...
🔧 Setting up VertexAI embeddings...
✅ Embeddings ready!
🔧 Setting up Pinecone vector store...
📊 Index stats: 1234 vectors
✅ Pinecone ready!

📝 Options:
1. Ask a question (RAG)
2. Similarity search  
3. Exit

🔍 Choose an option (1-3): 1

❓ Enter your question: What are the top companies in AI?

🤔 Thinking...

💡 Answer: Based on the available data, the top AI companies include...

📚 Sources used: 5 documents
📄 First source preview: Company XYZ is a leading AI company...
```

### Advanced RAG Features

#### Custom Embeddings Class

The module provides a LangChain-compatible embeddings wrapper:

```python
from src.utils.ml.embeddings.src.vertex_ai_embeddings import VertexAIEmbeddings

embeddings = VertexAIEmbeddings(
    project_id="your-project-id",
    location="europe-west1",
    model="text-embedding-005"
)

# Use with LangChain
from langchain.vectorstores import Pinecone
vectorstore = Pinecone.from_existing_index(
    index_name="your-index",
    embedding=embeddings
)
```

### Troubleshooting

#### Common Issues

1. **Migration fails**: Check BigQuery permissions and table schema
2. **RAG system won't start**: Verify environment variables and API keys
3. **Slow search**: Consider reducing vector dimensions or using filters
4. **Memory issues**: Reduce batch size in migration

#### Debug Mode

Enable debug logging:

```python
import logging
logging.basicConfig(level=logging.DEBUG)
```

### Next Steps

1. **Deploy to production**: Use the migration script to move your embeddings
2. **Build RAG applications**: Integrate with your existing systems
3. **Monitor performance**: Track search latency and accuracy
4. **Scale as needed**: Add more indexes for different data types

For more examples and advanced usage, check the `examples/` directory in the scripts folder.

