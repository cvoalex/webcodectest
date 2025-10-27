# Models Root Configuration Feature

## Overview
Added `models_root` configuration parameter to both inference and compositing servers to simplify model path management. This allows you to specify a base directory once and use relative paths for individual models, avoiding repetitive full paths.

## Benefits
- **Cleaner Configuration**: Avoid repeating the same base path for every model
- **Easier Management**: Change all model locations by updating one path
- **Portability**: Configurations are more portable across different environments
- **Backward Compatible**: Absolute paths still work - they're used as-is

## Configuration Examples

### Inference Server (`go-inference-server/config.yaml`)

#### Before (Absolute Paths)
```yaml
models:
  sanders:
    model_path: "d:/Projects/webcodecstest/minimal_server/models/sanders/checkpoint/model_best.onnx"
  user_123:
    model_path: "d:/Projects/webcodecstest/minimal_server/models/user_123/checkpoint/model_best.onnx"
  user_456:
    model_path: "d:/Projects/webcodecstest/minimal_server/models/user_456/checkpoint/model_best.onnx"
```

#### After (With models_root)
```yaml
models_root: "d:/Projects/webcodecstest/minimal_server/models"

models:
  sanders:
    model_path: "sanders/checkpoint/model_best.onnx"
  user_123:
    model_path: "user_123/checkpoint/model_best.onnx"
  user_456:
    model_path: "user_456/checkpoint/model_best.onnx"
```

### Compositing Server (`go-compositing-server/config.yaml`)

#### Before (Absolute Paths)
```yaml
models:
  sanders:
    background_dir: "d:/Projects/webcodecstest/minimal_server/models/sanders/frames"
    crop_rects_path: "d:/Projects/webcodecstest/minimal_server/models/sanders/crop_rects.json"
  user_123:
    background_dir: "d:/Projects/webcodecstest/minimal_server/models/user_123/frames"
    crop_rects_path: "d:/Projects/webcodecstest/minimal_server/models/user_123/crop_rects.json"
```

#### After (With models_root)
```yaml
models_root: "d:/Projects/webcodecstest/minimal_server/models"

models:
  sanders:
    background_dir: "sanders/frames"
    crop_rects_path: "sanders/crop_rects.json"
  user_123:
    background_dir: "user_123/frames"
    crop_rects_path: "user_123/crop_rects.json"
```

## Path Resolution Logic

### Inference Server
- Resolves `model_path` relative to `models_root` if:
  - `model_path` is not an absolute path (doesn't start with `/` or `C:\` etc.)
  - `models_root` is configured
- Logs the resolved path for debugging: `Model path: <resolved-path>`

### Compositing Server
- Resolves `background_dir` and `crop_rects_path` relative to `models_root` if:
  - Path is not absolute
  - `models_root` is configured
- Logs both resolved paths for debugging:
  - `Crop rects path: <resolved-path>`
  - `Background dir: <resolved-path>`

## Implementation Details

### Files Modified

#### Inference Server
1. **config/config.go**: Added `ModelsRoot string` field to Config struct
2. **registry/registry.go**: Added path resolution logic when loading models
   - Imports `path/filepath` package
   - Checks if path is absolute
   - Joins with `models_root` if relative

#### Compositing Server
1. **config/config.go**: Added `ModelsRoot string` field to Config struct
2. **registry/registry.go**: Added path resolution logic for backgrounds and crop rects
   - Imports `path/filepath` package
   - Resolves both `background_dir` and `crop_rects_path`
   - Logs resolved paths for verification

### Code Example (Path Resolution)
```go
// Resolve path relative to models_root if not absolute
modelPath := modelCfg.ModelPath
if !filepath.IsAbs(modelPath) && r.config.ModelsRoot != "" {
    modelPath = filepath.Join(r.config.ModelsRoot, modelPath)
}
log.Printf("   Model path: %s", modelPath)
```

## Usage Guidelines

1. **Set models_root once** at the top of your config file
2. **Use relative paths** for models in the same directory tree
3. **Use absolute paths** for models in different locations (e.g., external drives)
4. **Mix and match**: You can use both relative and absolute paths in the same config

## Example: Mixed Paths
```yaml
models_root: "d:/Projects/webcodecstest/minimal_server/models"

models:
  sanders:
    # Relative path - uses models_root
    model_path: "sanders/checkpoint/model_best.onnx"
    
  external_model:
    # Absolute path - used as-is
    model_path: "e:/ExternalStorage/models/special/model.onnx"
```

## Testing
Both servers compile successfully with the new feature:
```bash
# Inference server
cd go-inference-server
go build -o inference-server.exe ./cmd/server

# Compositing server  
cd go-compositing-server
go build -o compositing-server.exe ./cmd/server
```

No lint errors or compilation issues.

## Production Impact
- **No breaking changes**: Existing configs with absolute paths continue to work
- **Optional feature**: `models_root` is optional - omit it to use absolute paths
- **Cleaner configs**: Multi-tenant deployments with many models benefit most
- **Better portability**: Moving installations between servers is easier
