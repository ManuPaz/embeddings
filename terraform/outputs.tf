# Outputs for BigQuery embedding models Terraform configuration

output "bigquery_connection_name" {
  description = "The name of the BigQuery connection"
  value       = google_bigquery_connection.vertex_ai_connection.name
}

output "bigquery_connection_service_account" {
  description = "The service account ID for the BigQuery connection"
  value       = google_bigquery_connection.vertex_ai_connection.cloud_resource[0].service_account_id
}

output "bigquery_connection_id" {
  description = "The ID of the BigQuery connection"
  value       = google_bigquery_connection.vertex_ai_connection.connection_id
}

output "text_embedding_model_name" {
  description = "The name of the text embedding model"
  value       = var.text_embedding_model_name
}

output "text_embedding_gemini_model_name" {
  description = "The name of the text embedding Gemini model"
  value       = var.text_embedding_gemini_model_name
}

output "dataset_id" {
  description = "The BigQuery dataset ID where models are created"
  value       = var.dataset_id
}

output "project_id" {
  description = "The GCP project ID where resources are created"
  value       = var.project_id
}

output "location" {
  description = "The GCP location where resources are created"
  value       = var.location
}
