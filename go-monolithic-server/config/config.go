package config

import (
	"fmt"
	"os"
	"path/filepath"

	"gopkg.in/yaml.v3"
)

// Config represents the monolithic server configuration
type Config struct {
	Server     ServerConfig           `yaml:"server"`
	GPUs       GPUConfig              `yaml:"gpus"`
	Capacity   CapacityConfig         `yaml:"capacity"`
	ONNX       ONNXConfig             `yaml:"onnx"`
	Output     OutputConfig           `yaml:"output"`
	Logging    LoggingConfig          `yaml:"logging"`
	ModelsRoot string                 `yaml:"models_root"`
	Models     map[string]ModelConfig `yaml:"models"`
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
	MaxModels             int    `yaml:"max_models"`
	MaxMemoryGB           int    `yaml:"max_memory_gb"`
	BackgroundCacheFrames int    `yaml:"background_cache_frames"`
	EvictionPolicy        string `yaml:"eviction_policy"`
	IdleTimeoutMinutes    int    `yaml:"idle_timeout_minutes"`
}

type ONNXConfig struct {
	LibraryPath          string `yaml:"library_path"`
	CUDAStreamsPerWorker int    `yaml:"cuda_streams_per_worker"`
	IntraOpThreads       int    `yaml:"intra_op_threads"`
	InterOpThreads       int    `yaml:"inter_op_threads"`
}

type OutputConfig struct {
	Format      string `yaml:"format"`       // "jpeg" or "raw"
	JPEGQuality int    `yaml:"jpeg_quality"` // 1-100
}

type LoggingConfig struct {
	Level               string `yaml:"level"`
	LogInferenceTimes   bool   `yaml:"log_inference_times"`
	LogGPUUtilization   bool   `yaml:"log_gpu_utilization"`
	LogCompositingTimes bool   `yaml:"log_compositing_times"`
	LogCacheStats       bool   `yaml:"log_cache_stats"`
	BufferedLogging     bool   `yaml:"buffered_logging"`
	SampleRate          int    `yaml:"sample_rate"`
	AutoFlush           bool   `yaml:"auto_flush"`
	FlushIntervalMs     int    `yaml:"flush_interval_ms"`
}

type ModelConfig struct {
	ModelPath          string `yaml:"model_path"`
	BackgroundDir      string `yaml:"background_dir"`
	CropRectsPath      string `yaml:"crop_rects_path"`
	NumFrames          int    `yaml:"num_frames"`
	PreloadBackgrounds bool   `yaml:"preload_backgrounds"`
	PreloadModel       bool   `yaml:"preload_model"`
	PreferredGPU       int    `yaml:"preferred_gpu"` // -1 = auto
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
	if cfg.Capacity.BackgroundCacheFrames == 0 {
		cfg.Capacity.BackgroundCacheFrames = 600
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
	if cfg.Output.Format == "" {
		cfg.Output.Format = "jpeg"
	}
	if cfg.Output.JPEGQuality == 0 {
		cfg.Output.JPEGQuality = 75
	}
	if cfg.Logging.Level == "" {
		cfg.Logging.Level = "info"
	}

	// Resolve model paths relative to models_root if needed
	if cfg.ModelsRoot != "" {
		for modelID, modelCfg := range cfg.Models {
			// Resolve model_path if relative
			if !filepath.IsAbs(modelCfg.ModelPath) {
				modelCfg.ModelPath = filepath.Join(cfg.ModelsRoot, modelCfg.ModelPath)
			}
			// Resolve background_dir if relative
			if modelCfg.BackgroundDir != "" && !filepath.IsAbs(modelCfg.BackgroundDir) {
				modelCfg.BackgroundDir = filepath.Join(cfg.ModelsRoot, modelCfg.BackgroundDir)
			}
			// Resolve crop_rects_path if relative
			if modelCfg.CropRectsPath != "" && !filepath.IsAbs(modelCfg.CropRectsPath) {
				modelCfg.CropRectsPath = filepath.Join(cfg.ModelsRoot, modelCfg.CropRectsPath)
			}
			cfg.Models[modelID] = modelCfg
		}
	}

	return &cfg, nil
}
