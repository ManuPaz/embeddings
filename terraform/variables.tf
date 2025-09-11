# Variables for BigQuery embedding models Terraform configuration

variable "project_id" {
  description = "The GCP project ID where resources will be created"
  type        = string
}

variable "location" {
  description = "The GCP location/region for the resources"
  type        = string
  default     = "europe-west1"
}

variable "connection_id" {
  description = "The ID for the BigQuery connection to Vertex AI"
  type        = string
}

variable "dataset_id" {
  description = "The BigQuery dataset ID where models will be created"
  type        = string
}

variable "text_embedding_model_name" {
  description = "The name for the text embedding model"
  type        = string
}

variable "text_embedding_endpoint" {
  description = "The Vertex AI endpoint for text embedding"
  type        = string
}

variable "text_embedding_gemini_model_name" {
  description = "The name for the text embedding Gemini model"
  type        = string
}

variable "text_embedding_gemini_endpoint" {
  description = "The Vertex AI endpoint for text embedding Gemini"
  type        = string
}