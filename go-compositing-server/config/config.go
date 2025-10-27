package config

import (
	"fmt"
	"os"

	"gopkg.in/yaml.v3"
)

// Config represents the compositing server configuration
type Config struct {
	Server          ServerConfig           `yaml:"server"`
	InferenceServer InferenceServerConfig  `yaml:"inference_server"`
	Capacity        CapacityConfig         `yaml:"capacity"`
	Output          OutputConfig           `yaml:"output"`
	ModelsRoot      string                 `yaml:"models_root"` // Root directory for all models
	Models          map[string]ModelConfig `yaml:"models"`
	Logging         LoggingConfig          `yaml:"logging"`
}

type ServerConfig struct {
	Port             string `yaml:"port"`
	MaxMessageSizeMB int    `yaml:"max_message_size_mb"`
}

type InferenceServerConfig struct {
	URL            string `yaml:"url"`
	TimeoutSeconds int    `yaml:"timeout_seconds"`
	MaxRetries     int    `yaml:"max_retries"`
}

type CapacityConfig struct {
	MaxModels             int    `yaml:"max_models"`
	BackgroundCacheFrames int    `yaml:"background_cache_frames"`
	EvictionPolicy        string `yaml:"eviction_policy"`
	IdleTimeoutMinutes    int    `yaml:"idle_timeout_minutes"`
}

type OutputConfig struct {
	Format      string `yaml:"format"`       // "jpeg" or "raw"
	JpegQuality int    `yaml:"jpeg_quality"` // 1-100, default 85
}

type ModelConfig struct {
	ModelPath          string `yaml:"model_path"`
	BackgroundDir      string `yaml:"background_dir"`
	CropRectsPath      string `yaml:"crop_rects_path"`
	NumFrames          int    `yaml:"num_frames"`
	PreloadBackgrounds bool   `yaml:"preload_backgrounds"`
}

type LoggingConfig struct {
	Level               string `yaml:"level"`
	LogCompositingTimes bool   `yaml:"log_compositing_times"`
	LogCacheStats       bool   `yaml:"log_cache_stats"`
	BufferedLogging     bool   `yaml:"buffered_logging"`      // Use buffered logging (zero latency impact)
	SampleRate          int    `yaml:"sample_rate"`           // 0=log all, N=log 1 in N requests
	AutoFlush           bool   `yaml:"auto_flush"`            // Auto-flush logs every 100ms
	FlushIntervalMs     int    `yaml:"flush_interval_ms"`     // How often to flush (if auto_flush enabled)
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
	if cfg.InferenceServer.TimeoutSeconds == 0 {
		cfg.InferenceServer.TimeoutSeconds = 10
	}
	if cfg.InferenceServer.MaxRetries == 0 {
		cfg.InferenceServer.MaxRetries = 3
	}
	if cfg.Capacity.MaxModels == 0 {
		cfg.Capacity.MaxModels = 1000
	}
	if cfg.Capacity.BackgroundCacheFrames == 0 {
		cfg.Capacity.BackgroundCacheFrames = 50
	}
	if cfg.Capacity.EvictionPolicy == "" {
		cfg.Capacity.EvictionPolicy = "lfu"
	}
	if cfg.Output.Format == "" {
		cfg.Output.Format = "jpeg"
	}
	if cfg.Output.JpegQuality == 0 {
		cfg.Output.JpegQuality = 85
	}
	if cfg.Logging.Level == "" {
		cfg.Logging.Level = "info"
	}

	return &cfg, nil
}
