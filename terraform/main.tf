# ══════════════════════════════════════════
# Smart Campus Assistant — Infrastructure as Code
# Defines Supabase project resources declaratively
#
# Usage:
#   terraform init
#   terraform plan
#   terraform apply
# ══════════════════════════════════════════

terraform {
  required_version = ">= 1.5.0"

  required_providers {
    supabase = {
      source  = "supabase/supabase"
      version = "~> 1.0"
    }
  }
}

# ── Provider Configuration ──
provider "supabase" {
  access_token = var.supabase_access_token
}

# ── Variables ──
variable "supabase_access_token" {
  description = "Supabase personal access token"
  type        = string
  sensitive   = true
}

variable "project_name" {
  description = "Supabase project name"
  type        = string
  default     = "smart-campus-assistant"
}

variable "organization_id" {
  description = "Supabase organization ID"
  type        = string
}

variable "db_password" {
  description = "Database password for the Supabase project"
  type        = string
  sensitive   = true
}

# ── Supabase Project ──
resource "supabase_project" "campus_assistant" {
  organization_id   = var.organization_id
  name              = var.project_name
  database_password  = var.db_password
  region            = "eu-central-1"

  lifecycle {
    prevent_destroy = true
  }
}

# ── Outputs ──
output "project_id" {
  description = "Supabase project ID"
  value       = supabase_project.campus_assistant.id
}

output "project_url" {
  description = "Supabase project API URL"
  value       = "https://${supabase_project.campus_assistant.id}.supabase.co"
}

output "database_url" {
  description = "Direct PostgreSQL connection string (use with pgvector)"
  value       = "postgresql://postgres:${var.db_password}@db.${supabase_project.campus_assistant.id}.supabase.co:5432/postgres"
  sensitive   = true
}
