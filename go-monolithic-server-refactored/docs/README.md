# 📚 Go Monolithic Server Documentation Hub

> **Complete documentation for the high-performance lip-sync inference server**

Welcome to the documentation hub! This directory contains comprehensive documentation for understanding, maintaining, and extending the Go monolithic server.

---

## 🚀 Quick Navigation

### Getting Started
- **[Project Vision & Goals](../README.md#project-vision)** - What we're building and why
- **[Quick Start Guide](backend/development/DEVELOPMENT_GUIDE.md)** - Set up and run the server
- **[Architecture Overview](backend/ARCHITECTURE.md)** - System design and components

### Technical Documentation
- **[Architecture](backend/ARCHITECTURE.md)** - System design, components, data flow
- **[API Reference](backend/API_REFERENCE.md)** - All gRPC endpoints with examples
- **[Performance Optimizations](backend/ARCHITECTURE.md#performance-optimizations)** - What was optimized and why

### Development
- **[Testing Guide](backend/development/TESTING.md)** - Test coverage, how to run tests
- **[Common Gotchas](backend/development/GOTCHAS.md)** - Pitfalls and how to avoid them
- **[Development Guide](backend/development/DEVELOPMENT_GUIDE.md)** - How to contribute

### Decision Records
- **[ADR Index](backend/adr/)** - Why we made key architectural decisions
  - [ADR-001: Parallel Image Processing](backend/adr/ADR-001-parallel-image-processing.md)
  - [ADR-002: Memory Pooling Strategy](backend/adr/ADR-002-memory-pooling.md)
  - [ADR-003: Parallel Mel Extraction](backend/adr/ADR-003-parallel-mel-extraction.md)

### Session Notes
- **[Session Notes Index](backend/session-notes/)** - Development logs and bug fixes
  - [Phase 1 Optimization](backend/session-notes/2025-10-29-phase1-optimization.md)
  - [Phase 2 Mel Extraction](backend/session-notes/2025-10-29-phase2-mel-extraction.md)

---

## 📖 Documentation Categories

### 🏗️ Architecture
Understand how the system works, component relationships, and design patterns.

| Document | Purpose | Audience |
|----------|---------|----------|
| [ARCHITECTURE.md](backend/ARCHITECTURE.md) | System design, components, data flow | All developers |
| [Performance Optimizations](backend/ARCHITECTURE.md#performance-optimizations) | What we optimized and benchmarks | Performance engineers |

### 🔌 API
Learn how to use the server's gRPC API with concrete examples.

| Document | Purpose | Audience |
|----------|---------|----------|
| [API_REFERENCE.md](backend/API_REFERENCE.md) | All endpoints with request/response examples | API users, integrators |

### 🧪 Testing
Understand test coverage, how to run tests, and testing best practices.

| Document | Purpose | Audience |
|----------|---------|----------|
| [TESTING.md](backend/development/TESTING.md) | Test coverage, running tests, writing tests | All developers |

### 🎯 Decisions
Why we made key architectural and technical decisions.

| Document | Purpose | Audience |
|----------|---------|----------|
| [ADR Template](backend/adr/template.md) | How to write Architecture Decision Records | Contributors |
| [ADR-001](backend/adr/ADR-001-parallel-image-processing.md) | Parallel image processing (4-5x speedup) | All developers |
| [ADR-002](backend/adr/ADR-002-memory-pooling.md) | Memory pooling strategy (1000x reduction) | Performance engineers |
| [ADR-003](backend/adr/ADR-003-parallel-mel-extraction.md) | Parallel mel extraction (1.5x speedup) | Audio engineers |

### 📝 Session Notes
Development logs documenting fixes, optimizations, and major changes.

| Document | Purpose | Audience |
|----------|---------|----------|
| [2025-10-29: Phase 1](backend/session-notes/2025-10-29-phase1-optimization.md) | Image processing optimization | All developers |
| [2025-10-29: Phase 2](backend/session-notes/2025-10-29-phase2-mel-extraction.md) | Audio mel extraction optimization | Audio engineers |

### 🔧 Development
Practical guides for developing, testing, and contributing.

| Document | Purpose | Audience |
|----------|---------|----------|
| [DEVELOPMENT_GUIDE.md](backend/development/DEVELOPMENT_GUIDE.md) | How to set up and contribute | New developers |
| [GOTCHAS.md](backend/development/GOTCHAS.md) | Common mistakes and pitfalls | All developers |

---

## 🎯 Use Cases

### "I'm new to this project"
1. Start with [README.md](../README.md) - Understand the vision
2. Read [ARCHITECTURE.md](backend/ARCHITECTURE.md) - Learn system design
3. Follow [DEVELOPMENT_GUIDE.md](backend/development/DEVELOPMENT_GUIDE.md) - Set up your environment
4. Review [GOTCHAS.md](backend/development/GOTCHAS.md) - Avoid common mistakes

### "I need to use the API"
1. Read [API_REFERENCE.md](backend/API_REFERENCE.md) - All endpoints
2. Try the examples - Copy/paste and test
3. Check [GOTCHAS.md](backend/development/GOTCHAS.md) - API-specific pitfalls

### "I want to understand why something works this way"
1. Check [ADR Index](backend/adr/) - Find relevant decision record
2. Read the ADR - Understand context, alternatives, consequences
3. Check [Session Notes](backend/session-notes/) - See what changed over time

### "I need to optimize performance"
1. Read [Performance Optimizations](backend/ARCHITECTURE.md#performance-optimizations)
2. Review [ADR-001](backend/adr/ADR-001-parallel-image-processing.md) and [ADR-002](backend/adr/ADR-002-memory-pooling.md)
3. Check [TESTING.md](backend/development/TESTING.md) - Run benchmarks

### "I'm fixing a bug"
1. Check [GOTCHAS.md](backend/development/GOTCHAS.md) - Known issues
2. Document your fix in [session-notes/](backend/session-notes/)
3. Add tests to prevent regression

---

## ⚠️ Documentation Maintenance

**CRITICAL: Documentation must be kept up-to-date as the project evolves!**

### When to Update Documentation

| Change | Update These Documents |
|--------|----------------------|
| Architecture changes | ARCHITECTURE.md, relevant ADRs |
| New features | ARCHITECTURE.md, API_REFERENCE.md, TESTING.md |
| Performance changes | ARCHITECTURE.md, ADRs, session notes |
| Bug fixes | Session notes, GOTCHAS.md |
| Major decisions | Write new ADR |
| API changes | API_REFERENCE.md |

### Documentation Checklist for PRs

Before merging any PR, ensure:

- [ ] Updated relevant architecture docs
- [ ] Added/updated API documentation if API changed
- [ ] Updated performance metrics if optimizations made
- [ ] Created ADR for significant architectural decisions
- [ ] Updated GOTCHAS.md if new pitfalls discovered
- [ ] Updated TESTING.md if test coverage changed
- [ ] Added session note documenting major changes
- [ ] Updated README if user-facing changes

### How to Write Documentation

**Follow these principles:**

1. **Documentation-First** - Code is temporary, knowledge is permanent
2. **Show, Don't Tell** - Include examples, not just descriptions
3. **Cross-Reference** - Link related documents together
4. **Keep Current** - Update docs with every significant change
5. **Think Future** - Write for someone with zero context
6. **Be Comprehensive** - Cover "why" not just "what"
7. **Make Actionable** - Provide copy/paste examples

**Templates available:**
- [ADR Template](backend/adr/template.md) - For architectural decisions
- [Session Note Template](backend/session-notes/template.md) - For development logs

---

## 📊 Documentation Status

### Coverage Summary

| Category | Documents | Status |
|----------|-----------|--------|
| **Architecture** | 1 | ✅ Complete |
| **API Reference** | 1 | ✅ Complete |
| **ADRs** | 3 | ✅ Complete |
| **Testing** | 1 | ✅ Complete |
| **Development Guides** | 2 | ✅ Complete |
| **Session Notes** | 2 | ✅ Complete |
| **TOTAL** | **10** | **✅ 100%** |

### Quality Metrics

- ✅ All API endpoints documented with examples
- ✅ All major architectural decisions have ADRs
- ✅ All 35 functional tests documented
- ✅ Common gotchas documented
- ✅ Performance benchmarks included
- ✅ Cross-references work
- ✅ Examples are copy-pasteable

---

## 🤝 Contributing to Documentation

### Adding New Documentation

1. **Choose the right location:**
   - Backend architecture → `backend/`
   - API changes → `backend/API_REFERENCE.md`
   - Decisions → `backend/adr/`
   - Development logs → `backend/session-notes/`

2. **Use templates:**
   - [ADR Template](backend/adr/template.md)
   - [Session Note Template](backend/session-notes/template.md)

3. **Follow format:**
   - Clear headings
   - Code examples
   - ASCII diagrams where helpful
   - Cross-references to related docs

4. **Update this hub:**
   - Add link to new document
   - Update status table

### Improving Existing Documentation

1. Fix inaccuracies immediately
2. Add missing examples
3. Improve clarity
4. Update benchmarks
5. Add cross-references

### Archiving Old Documentation

When deprecating features:
1. Move docs to `/archive`
2. Update archive README with reason
3. Remove from this hub
4. Add redirect note in old location

---

## 📚 Related Resources

- **[Go Documentation](https://go.dev/doc/)** - Official Go docs
- **[gRPC Go](https://grpc.io/docs/languages/go/)** - gRPC in Go
- **[ONNX Runtime](https://onnxruntime.ai/)** - ONNX inference
- **[Architecture Decision Records](https://adr.github.io/)** - ADR best practices
- **[Write the Docs](https://www.writethedocs.org/)** - Documentation community

---

## 🎯 Success Criteria

Documentation is considered complete when:

✅ **A new developer can:**
- Understand the project vision in 5 minutes
- Set up their environment in 30 minutes
- Make their first contribution in 1 day
- Navigate the codebase confidently

✅ **An AI assistant can:**
- Understand system architecture
- Know why decisions were made
- Find relevant code quickly
- Suggest changes that align with patterns

✅ **The team can:**
- Onboard new members efficiently
- Reference decisions made months ago
- Avoid repeating past mistakes
- Maintain consistency across changes

✅ **Documentation is:**
- Comprehensive (covers everything)
- Current (reflects actual code)
- Cross-referenced (easy to navigate)
- Actionable (includes examples)
- Maintainable (clear update process)

---

**Last Updated:** November 6, 2025  
**Documentation Version:** 1.0.0  
**Project Version:** Phase 2 Complete (48 FPS target achieved)

---

**Questions? Issues?**  
- Check [GOTCHAS.md](backend/development/GOTCHAS.md) for common problems
- Review [ADRs](backend/adr/) for decision context
- See [Session Notes](backend/session-notes/) for recent changes
