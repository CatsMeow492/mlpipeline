-- Initialize PostgreSQL databases for ML Pipeline

-- Create MLflow database
CREATE DATABASE mlflow;

-- Create additional schemas for pipeline data
\c mlpipeline;

CREATE SCHEMA IF NOT EXISTS experiments;
CREATE SCHEMA IF NOT EXISTS monitoring;
CREATE SCHEMA IF NOT EXISTS metadata;

-- Create tables for experiment tracking
CREATE TABLE IF NOT EXISTS experiments.pipeline_runs (
    id SERIAL PRIMARY KEY,
    experiment_id VARCHAR(255) NOT NULL,
    run_id VARCHAR(255) UNIQUE NOT NULL,
    name VARCHAR(255),
    status VARCHAR(50),
    start_time TIMESTAMP,
    end_time TIMESTAMP,
    config_hash VARCHAR(64),
    git_commit VARCHAR(40),
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

-- Create tables for monitoring data
CREATE TABLE IF NOT EXISTS monitoring.drift_reports (
    id SERIAL PRIMARY KEY,
    experiment_id VARCHAR(255),
    timestamp TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    data_drift_score FLOAT,
    prediction_drift_score FLOAT,
    drift_detected BOOLEAN,
    report_path VARCHAR(500)
);

-- Create tables for metadata
CREATE TABLE IF NOT EXISTS metadata.data_versions (
    id SERIAL PRIMARY KEY,
    dataset_name VARCHAR(255) NOT NULL,
    version VARCHAR(50) NOT NULL,
    hash VARCHAR(64) NOT NULL,
    size_bytes BIGINT,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    UNIQUE(dataset_name, version)
);

-- Create indexes for performance
CREATE INDEX IF NOT EXISTS idx_pipeline_runs_experiment_id ON experiments.pipeline_runs(experiment_id);
CREATE INDEX IF NOT EXISTS idx_pipeline_runs_status ON experiments.pipeline_runs(status);
CREATE INDEX IF NOT EXISTS idx_drift_reports_timestamp ON monitoring.drift_reports(timestamp);
CREATE INDEX IF NOT EXISTS idx_data_versions_dataset ON metadata.data_versions(dataset_name);

-- Grant permissions
GRANT ALL PRIVILEGES ON ALL TABLES IN SCHEMA experiments TO mlpipeline;
GRANT ALL PRIVILEGES ON ALL TABLES IN SCHEMA monitoring TO mlpipeline;
GRANT ALL PRIVILEGES ON ALL TABLES IN SCHEMA metadata TO mlpipeline;
GRANT ALL PRIVILEGES ON ALL SEQUENCES IN SCHEMA experiments TO mlpipeline;
GRANT ALL PRIVILEGES ON ALL SEQUENCES IN SCHEMA monitoring TO mlpipeline;
GRANT ALL PRIVILEGES ON ALL SEQUENCES IN SCHEMA metadata TO mlpipeline;