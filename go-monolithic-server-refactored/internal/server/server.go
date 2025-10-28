package server

import (
	"go-monolithic-server/audio"
	"go-monolithic-server/config"
	"go-monolithic-server/internal/compositing"
	"go-monolithic-server/logger"
	pb "go-monolithic-server/proto"
	"go-monolithic-server/registry"
)

// Server holds all dependencies for the monolithic lip-sync server
type Server struct {
	pb.UnimplementedMonolithicServiceServer

	// Core dependencies
	modelRegistry    *registry.ModelRegistry
	imageRegistry    *registry.ImageRegistry
	cfg              *config.Config

	// Audio processing
	audioProcessor   *audio.Processor
	audioEncoderPool *audio.AudioEncoderPool

	// Compositing
	compositor *compositing.Compositor

	// Logging
	logger *logger.BufferedLogger
}

// New creates a new monolithic server with all dependencies initialized
func New(
	modelRegistry *registry.ModelRegistry,
	imageRegistry *registry.ImageRegistry,
	cfg *config.Config,
	audioProcessor *audio.Processor,
	audioEncoderPool *audio.AudioEncoderPool,
	compositor *compositing.Compositor,
	logger *logger.BufferedLogger,
) *Server {
	return &Server{
		modelRegistry:    modelRegistry,
		imageRegistry:    imageRegistry,
		cfg:              cfg,
		audioProcessor:   audioProcessor,
		audioEncoderPool: audioEncoderPool,
		compositor:       compositor,
		logger:           logger,
	}
}
