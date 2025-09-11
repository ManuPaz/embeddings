# BigQuery Embedding Models - Terraform Configuration

This directory contains Terraform configuration files to automate the creation of BigQuery embedding models for semantic search.

## Overview

The Terraform configuration creates:
- BigQuery connection to Vertex AI
- IAM permissions for the connection service account
- BigQuery models for different embedding types:
  - `text_embedding` (text-embedding-005)
  - `text_embedding_gemini` (text-embedding-005)
- Uses BigQuery jobs to execute CREATE MODEL statements

## Files Structure

```
terraform/
├── main.tf                 # Main Terraform configuration
├── variables.tf            # Variable definitions
├── terraform.tfvars        # Real values (production)
├── terraform.tfvars.example # Example values (template)
├── README.md               # This file
└── .gitignore              # Git ignore file
```

## Prerequisites

1. **Terraform**: Version 1.0 or higher
2. **Google Cloud SDK**: Authenticated with appropriate permissions
3. **GCP Permissions**: 
   - BigQuery Admin
   - AI Platform User
   - Service Account Admin (if creating new service accounts)

## Quick Start

### 1. Initialize Terraform

```bash
cd src/utils/ml/embeddings/terraform
terraform init
```

### 2. Configure Variables

Copy the example file and update with your values:

```bash
cp terraform.tfvars.example terraform.tfvars
# Edit terraform.tfvars with your real values
```

### 3. Review the Plan

```bash
terraform plan
```

### 4. Apply the Configuration

```bash
terraform apply
```

## Configuration Options

### Required Variables

- `project_id`: Your GCP project ID
- `connection_id`: BigQuery connection ID for Vertex AI
- `dataset_id`: BigQuery dataset ID for models
- `text_embedding_model_name`: Name for text embedding model
- `text_embedding_endpoint`: Vertex AI endpoint for text embedding

### Optional Variables

- `location`: GCP region (default: europe-west1)
- `create_dataset`: Whether to create the dataset (default: false)
- `dataset_labels`: Custom labels for the dataset

## Outputs

After successful execution, Terraform will output:

- `bigquery_connection_name`: The created connection name
- `bigquery_connection_service_account`: Service account ID
- `text_embedding_model_name`: Text embedding model name
- `text_embedding_gemini_model_name`: Gemini model name
- `dataset_id`: BigQuery dataset ID
- `project_id`: GCP project ID
- `location`: GCP location

## Manual Commands Equivalent

This Terraform configuration replaces these manual commands:

```bash
# Create BigQuery connection
bq mk --connection --connection_type=CLOUD_RESOURCE \
  --project_id=YOUR_PROJECT --location=YOUR_LOCATION \
  YOUR_CONNECTION_ID

# Grant IAM permissions
gcloud projects add-iam-policy-binding YOUR_PROJECT \
  --member="serviceAccount:YOUR_SERVICE_ACCOUNT" \
  --role="roles/aiplatform.user"

# Create models (these are now automated by Terraform)
```

## Security Considerations

- The configuration creates a BigQuery connection with `roles/aiplatform.user`
- Service account credentials are managed by Google Cloud
- All resource names are parameterized to avoid hardcoding
- IAM permissions follow the principle of least privilege

## Troubleshooting

### Common Issues

1. **Permission Denied**: Ensure you have the required GCP roles
2. **Connection Failed**: Verify Vertex AI API is enabled
3. **Model Creation Error**: Check the endpoint names are correct

### Debug Commands

```bash
# Check Terraform state
terraform show

# Validate configuration
terraform validate

# Check GCP resources
gcloud bigquery connections list --project=YOUR_PROJECT
gcloud bigquery routines list --dataset=YOUR_DATASET --project=YOUR_PROJECT
```

## Maintenance

### Updating Models

To update model endpoints or configurations:

1. Modify the variables in `terraform.tfvars`
2. Run `terraform plan` to see changes
3. Run `terraform apply` to apply updates

### Destroying Resources

⚠️ **Warning**: This will remove all created resources

```bash
terraform destroy
```

## Contributing

When modifying this configuration:

1. Update both `terraform.tfvars` and `terraform.tfvars.example`
2. Test with `terraform plan` before applying
3. Update this README if adding new features
4. Follow Terraform best practices for naming and structure
