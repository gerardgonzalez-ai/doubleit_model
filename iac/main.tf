terraform {
  required_providers {
    google = {
      source  = "hashicorp/google"
      version = "3.5.0"
    }
  }
}

provider "google" {
  credentials = jsondecode(var.credentials)
  project     = var.project_id
  region      = var.region
}

resource "google_cloud_run_service" "doubletap_service" {
  name     = "doubletap-service"
  location = var.region

  template {
    spec {
      containers {
        image = "gerardgonzalez7/doubletap:latest"
      }
    }
  }

  traffic {
    percent         = 100
    latest_revision = true
  }

  autogenerate_revision_name = true
}

output "service_url" {
  value     = google_cloud_run_service.doubletap_service.status[0].url
  sensitive = false
}

variable "credentials_file" {}
variable "project_id" {
  default = "gg-website"
}
variable "region" {
  default = "us-central1"
}
