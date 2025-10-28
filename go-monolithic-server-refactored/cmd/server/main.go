package main

import (
	"fmt"
	"log"
	"net"
	"time"

	"google.golang.org/grpc"
	"google.golang.org/grpc/keepalive"

	"go-monolithic-server/audio"
	"go-monolithic-server/config"
	"go-monolithic-server/internal/compositing"
	"go-monolithic-server/internal/server"
	"go-monolithic-server/logger"
	pb "go-monolithic-server/proto"
	"go-monolithic-server/registry"
)

func main() {
	fmt.Println("================================================================================")
	fmt.Println("🚀 Monolithic Lipsync Server (Refactored)")
	fmt.Println("================================================================================")

	// Load configuration
	cfgPath := "config.yaml"
	cfg, err := config.Load(cfgPath)
	if err != nil {
		log.Fatalf("❌ Failed to load config: %v", err)
	}

	log.Printf("✅ Configuration loaded from %s", cfgPath)
	log.Printf("   GPUs: %d × %dGB", cfg.GPUs.Count, cfg.GPUs.MemoryGBPerGPU)
	log.Printf("   Workers per GPU: %d (total: %d workers)",
		cfg.Server.WorkerCountPerGPU, cfg.GPUs.Count*cfg.Server.WorkerCountPerGPU)
	log.Printf("   Max models: %d", cfg.Capacity.MaxModels)
	log.Printf("   Max memory: %d GB", cfg.Capacity.MaxMemoryGB)
	log.Printf("   Background cache: %d frames per model", cfg.Capacity.BackgroundCacheFrames)
	log.Printf("   Eviction policy: %s", cfg.Capacity.EvictionPolicy)
	log.Printf("   Configured models: %d", len(cfg.Models))

	// Create model registry (for ONNX models)
	log.Println("\n📦 Initializing model registry...")
	modelReg, err := registry.NewModelRegistry(cfg)
	if err != nil {
		log.Fatalf("❌ Failed to create model registry: %v", err)
	}
	defer modelReg.Close()

	log.Printf("✅ Model registry initialized (%d models preloaded)", modelReg.GetLoadedCount())

	// Create image registry (for backgrounds and crop rects)
	log.Println("\n🖼️  Initializing image registry...")
	imageReg, err := registry.NewImageRegistry(cfg)
	if err != nil {
		log.Fatalf("❌ Failed to create image registry: %v", err)
	}
	defer imageReg.Close()

	log.Printf("✅ Image registry initialized (%d models loaded)", imageReg.GetLoadedCount())

	// Display GPU info
	gpuInfo := modelReg.GetGPUInfo()
	fmt.Println("\n🎮 GPU Status:")
	for _, gpu := range gpuInfo {
		fmt.Printf("   GPU %d: %d models, %d MB used / %d MB total\n",
			gpu.GPUID,
			gpu.LoadedModels,
			gpu.UsedMemory/(1024*1024),
			gpu.TotalMemory/(1024*1024))
	}

	// Initialize audio processing pipeline
	log.Println("\n🎵 Initializing audio processing pipeline...")
	audioProcessor := audio.NewProcessor(nil)
	log.Println("✅ Mel-spectrogram processor initialized")

	// Create audio encoder pool (4 instances for parallel processing)
	encoderPoolSize := 4
	audioEncoderPool, err := audio.NewAudioEncoderPool(encoderPoolSize, cfg.ONNX.LibraryPath)
	if err != nil {
		log.Printf("⚠️  Warning: Audio encoder pool not available: %v", err)
		log.Println("   Server will require pre-computed audio features")
		audioEncoderPool = nil
	} else {
		log.Printf("✅ Audio encoder pool initialized (%d instances for parallel processing)", encoderPoolSize)
	}

	// Create compositor
	compositor := compositing.NewCompositor(cfg.Output.JPEGQuality)
	log.Printf("✅ Compositor initialized (JPEG quality: %d)", cfg.Output.JPEGQuality)

	// Create buffered logger based on config
	var bufferedLog *logger.BufferedLogger
	if cfg.Logging.BufferedLogging {
		bufferedLog = logger.NewBufferedLogger(cfg.Logging.AutoFlush, cfg.Logging.SampleRate)
		defer bufferedLog.Stop()
		log.Printf("✅ Buffered logging enabled (sample_rate=%d, auto_flush=%v)",
			cfg.Logging.SampleRate, cfg.Logging.AutoFlush)
	}

	// Create server instance with all dependencies
	srv := server.New(
		modelReg,
		imageReg,
		cfg,
		audioProcessor,
		audioEncoderPool,
		compositor,
		bufferedLog,
	)

	// Create gRPC server with keep-alive
	maxSize := cfg.Server.MaxMessageSizeMB * 1024 * 1024

	var kaep = keepalive.EnforcementPolicy{
		MinTime:             5 * time.Second,
		PermitWithoutStream: true,
	}

	var kasp = keepalive.ServerParameters{
		Time:    10 * time.Second,
		Timeout: 3 * time.Second,
	}

	grpcServer := grpc.NewServer(
		grpc.MaxRecvMsgSize(maxSize),
		grpc.MaxSendMsgSize(maxSize),
		grpc.KeepaliveEnforcementPolicy(kaep),
		grpc.KeepaliveParams(kasp),
	)

	pb.RegisterMonolithicServiceServer(grpcServer, srv)

	// Start listening
	lis, err := net.Listen("tcp", cfg.Server.Port)
	if err != nil {
		log.Fatalf("❌ Failed to listen: %v", err)
	}

	fmt.Printf("\n🌐 Monolithic server listening on port %s\n", cfg.Server.Port)
	fmt.Println("   Protocol: gRPC with Protobuf")
	fmt.Println("   Features:")
	fmt.Println("      • Inference + Compositing in single process")
	fmt.Println("      • No inter-service communication overhead")
	fmt.Println("      • Multi-GPU inference")
	fmt.Println("      • Real-time audio processing (mel + encoder)")
	fmt.Println("      • Dynamic model loading")
	fmt.Println("      • Automatic eviction (LFU)")
	fmt.Println("      • JPEG-encoded output")
	fmt.Println("      • Refactored architecture for maintainability")
	fmt.Println("\n✅ Ready to accept connections!")
	fmt.Println("================================================================================")

	if err := grpcServer.Serve(lis); err != nil {
		log.Fatalf("❌ Failed to serve: %v", err)
	}
}
