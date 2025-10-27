package config

import (
	"fmt"
	"os"

	"gopkg.in/yaml.v3"
)

// Config represents the server configuration
type Config struct {
	Server     ServerConfig           `yaml:"server"`
	GPUs       GPUConfig              `yaml:"gpus"`
	Capacity   CapacityConfig         `yaml:"capacity"`
	ONNX       ONNXConfig             `yaml:"onnx"`
	ModelsRoot string                 `yaml:"models_root"` // Root directory for all models
	Models     map[string]ModelConfig `yaml:"models"`
	Logging    LoggingConfig          `yaml:"logging"`
}

type ServerConfig struct {
	Port              string `yaml:"port"`
	MaxMessageSizeMB  int    `yaml:"max_message_size_mb"`
	WorkerCountPerGPU int    `yaml:"worker_count_per_gpu"`
	QueueSize         int    `yaml:"queue_size"`
}

type GPUConfig struct {
	Enabled             bool   `yaml:"enabled"`
	Count               int    `yaml:"count"`
	MemoryGBPerGPU      int    `yaml:"memory_gb_per_gpu"`
	AssignmentStrategy  string `yaml:"assignment_strategy"`
	AllowMultiGPUModels bool   `yaml:"allow_multi_gpu_models"`
}

type CapacityConfig struct {
	MaxModels          int    `yaml:"max_models"`
	MaxMemoryGB        int    `yaml:"max_memory_gb"`
	EvictionPolicy     string `yaml:"eviction_policy"`
	IdleTimeoutMinutes int    `yaml:"idle_timeout_minutes"`
}

type ONNXConfig struct {
	LibraryPath          string `yaml:"library_path"`
	CUDAStreamsPerWorker int    `yaml:"cuda_streams_per_worker"`
	IntraOpThreads       int    `yaml:"intra_op_threads"`
	InterOpThreads       int    `yaml:"inter_op_threads"`
}

type ModelConfig struct {
	ModelPath    string `yaml:"model_path"`
	Preload      bool   `yaml:"preload"`
	PreferredGPU int    `yaml:"preferred_gpu"` // -1 = auto, 0-7 = specific GPU
}

type LoggingConfig struct {
	Level             string `yaml:"level"`
	LogInferenceTimes bool   `yaml:"log_inference_times"`
	LogGPUUtilization bool   `yaml:"log_gpu_utilization"`
}

// Load reads configuration from a YAML file
func Load(path string) (*Config, error) {
	data, err := os.ReadFile(path)
	if err != nil {
		return nil, fmt.Errorf("failed to read config file: %w", err)
	}

	var cfg Config
	if err := yaml.Unmarshal(data, &cfg); err != nil {
		return nil, fmt.Errorf("failed to parse config: %w", err)
	}

	// Set defaults
	if cfg.Server.MaxMessageSizeMB == 0 {
		cfg.Server.MaxMessageSizeMB = 50
	}
	if cfg.Server.WorkerCountPerGPU == 0 {
		cfg.Server.WorkerCountPerGPU = 4
	}
	if cfg.Server.QueueSize == 0 {
		cfg.Server.QueueSize = 200
	}
	if cfg.GPUs.Count == 0 {
		cfg.GPUs.Count = 1
	}
	if cfg.GPUs.AssignmentStrategy == "" {
		cfg.GPUs.AssignmentStrategy = "round-robin"
	}
	if cfg.Capacity.MaxModels == 0 {
		cfg.Capacity.MaxModels = 100
	}
	if cfg.Capacity.EvictionPolicy == "" {
		cfg.Capacity.EvictionPolicy = "lfu"
	}
	if cfg.ONNX.CUDAStreamsPerWorker == 0 {
		cfg.ONNX.CUDAStreamsPerWorker = 2
	}
	if cfg.ONNX.IntraOpThreads == 0 {
		cfg.ONNX.IntraOpThreads = 4
	}
	if cfg.ONNX.InterOpThreads == 0 {
		cfg.ONNX.InterOpThreads = 2
	}
	if cfg.Logging.Level == "" {
		cfg.Logging.Level = "info"
	}

	return &cfg, nil
}
