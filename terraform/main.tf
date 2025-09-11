# Terraform configuration for BigQuery embedding models
# This file creates the necessary infrastructure for embedding models in BigQuery

terraform {
  required_version = ">= 1.0"
  required_providers {
    google = {
      source  = "hashicorp/google"
      version = "~> 4.0"
    }
  }
}

# Create BigQuery connection for Vertex AI
resource "google_bigquery_connection" "vertex_ai_connection" {
  connection_id = var.connection_id
  project       = var.project_id
  location      = var.location

  cloud_resource {}
}

# Grant IAM permissions to the BigQuery connection service account
resource "google_project_iam_member" "bigquery_connection_aiplatform_user" {
  project = var.project_id
  role    = "roles/aiplatform.user"
  member  = "serviceAccount:${google_bigquery_connection.vertex_ai_connection.cloud_resource[0].service_account_id}"
}

# Create BigQuery models for different embedding types
resource "google_bigquery_job" "text_embedding_model" {
  project = var.project_id
  location = var.location
  job_id   = "create-${var.text_embedding_model_name}-${formatdate("YYYYMMDD-hhmmss", timestamp())}"

  query {
    query = <<-EOF
      CREATE OR REPLACE MODEL `${var.project_id}.${var.dataset_id}.${var.text_embedding_model_name}`
      REMOTE WITH CONNECTION `${var.project_id}.${var.location}.${var.connection_id}`
      OPTIONS(ENDPOINT = '${var.text_embedding_endpoint}')
    EOF

    create_disposition = ""
    write_disposition = ""
  }

  depends_on = [google_project_iam_member.bigquery_connection_aiplatform_user]
}

resource "google_bigquery_job" "text_embedding_gemini_model" {
  project = var.project_id
  location = var.location
  job_id   = "create-${var.text_embedding_gemini_model_name}-${formatdate("YYYYMMDD-hhmmss", timestamp())}"

  query {
    query = <<-EOF
      CREATE OR REPLACE MODEL `${var.project_id}.${var.dataset_id}.${var.text_embedding_gemini_model_name}`
      REMOTE WITH CONNECTION `${var.project_id}.${var.location}.${var.connection_id}`
      OPTIONS(ENDPOINT = '${var.text_embedding_gemini_endpoint}')
    EOF

    create_disposition = ""
    write_disposition = ""
  }

  depends_on = [google_project_iam_member.bigquery_connection_aiplatform_user]
}

