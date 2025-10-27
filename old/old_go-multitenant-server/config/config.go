package config

import (
	"fmt"
	"os"

	"gopkg.in/yaml.v3"
)

// Config is the main configuration structure
type Config struct {
	Server   ServerConfig           `yaml:"server"`
	Capacity CapacityConfig         `yaml:"capacity"`
	ONNX     ONNXConfig             `yaml:"onnx"`
	Models   map[string]ModelConfig `yaml:"models"`
	Logging  LoggingConfig          `yaml:"logging"`
}

type ServerConfig struct {
	Port             string `yaml:"port"`
	MaxMessageSizeMB int    `yaml:"max_message_size_mb"`
	WorkerCount      int    `yaml:"worker_count"` // Parallel inference workers
	QueueSize        int    `yaml:"queue_size"`   // Request queue size
}

type CapacityConfig struct {
	MaxModels             int    `yaml:"max_models"`
	MaxMemoryGB           int    `yaml:"max_memory_gb"`
	EvictionPolicy        string `yaml:"eviction_policy"` // "lru" or "lfu"
	IdleUnloadMinutes     int    `yaml:"idle_unload_minutes"`
	BackgroundCacheFrames int    `yaml:"background_cache_frames"` // Frames to cache per model
}

type ONNXConfig struct {
	LibraryPath string `yaml:"library_path"`
	CUDAEnabled bool   `yaml:"cuda_enabled"`
}

type ModelConfig struct {
	ModelPath        string `yaml:"model_path"`
	BackgroundDir    string `yaml:"background_dir"`
	CropRectsPath    string `yaml:"crop_rects_path"`
	NumFrames        int    `yaml:"num_frames"`
	Preload          bool   `yaml:"preload"`
	MemoryEstimateMB int    `yaml:"memory_estimate_mb"`
}

type LoggingConfig struct {
	Level              string `yaml:"level"`
	LogModelOperations bool   `yaml:"log_model_operations"`
	StatsReportMinutes int    `yaml:"stats_report_minutes"`
	LogFile            string `yaml:"log_file"`
}

// Load reads and parses the configuration file
func Load(path string) (*Config, error) {
	data, err := os.ReadFile(path)
	if err != nil {
		return nil, fmt.Errorf("failed to read config file: %w", err)
	}

	var cfg Config
	if err := yaml.Unmarshal(data, &cfg); err != nil {
		return nil, fmt.Errorf("failed to parse config file: %w", err)
	}

	// Validate configuration
	if err := cfg.Validate(); err != nil {
		return nil, fmt.Errorf("invalid configuration: %w", err)
	}

	return &cfg, nil
}

// Validate checks if the configuration is valid
func (c *Config) Validate() error {
	if c.Server.Port == "" {
		return fmt.Errorf("server.port is required")
	}

	if c.Capacity.MaxModels < 1 {
		return fmt.Errorf("capacity.max_models must be at least 1")
	}

	if c.Capacity.EvictionPolicy != "lru" && c.Capacity.EvictionPolicy != "lfu" {
		return fmt.Errorf("capacity.eviction_policy must be 'lru' or 'lfu'")
	}

	if len(c.Models) == 0 {
		return fmt.Errorf("at least one model must be configured")
	}

	// Validate each model
	for modelID, model := range c.Models {
		if model.ModelPath == "" {
			return fmt.Errorf("model '%s': model_path is required", modelID)
		}
		if model.BackgroundDir == "" {
			return fmt.Errorf("model '%s': background_dir is required", modelID)
		}
		if model.CropRectsPath == "" {
			return fmt.Errorf("model '%s': crop_rects_path is required", modelID)
		}
		if model.NumFrames < 1 {
			return fmt.Errorf("model '%s': num_frames must be at least 1", modelID)
		}
	}

	return nil
}
