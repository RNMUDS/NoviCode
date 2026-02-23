# RNNR_Coding

**Curriculum-Constrained Local AI Coding Agent for Education and Research**

RNNR_Coding is an open-source, offline-first coding agent that runs entirely on local hardware using [Ollama](https://ollama.ai). Unlike general-purpose coding assistants, RNNR_Coding is deliberately constrained to a fixed educational curriculum. This constraint is not a limitation — it is the core design principle.

---

## Why Constraints Matter

General-purpose coding agents suffer from a fundamental problem in educational settings: they generate anything the user asks for, across any language, framework, or domain. This produces several failure modes:

1. **Language mixing** — An LLM asked to "make a visualization" may produce Python, JavaScript, or a hybrid. Students receive inconsistent output that cannot be reliably executed in their environment.
2. **Scope drift** — Students accidentally receive code using advanced libraries, patterns, or languages they haven't learned yet.
3. **Unreproducible sessions** — Without structured logging, educators cannot analyze how students interact with AI tools.
4. **Security risks** — Unrestricted agents can install packages, make network requests, or modify system files.

RNNR_Coding eliminates these problems through strict mode isolation, language enforcement, and a validation layer that inspects every generated artifact before delivery.

---

## Supported Domains

| Mode | Domain | Language | Key Libraries |
|------|--------|----------|---------------|
| `python_basic` | Python fundamentals | Python | Standard library only |
| `py5` | Creative coding | Python | py5 (Processing) |
| `sklearn` | Machine learning basics | Python | scikit-learn, numpy |
| `pandas` | Data analysis | Python | pandas, matplotlib, seaborn |
| `aframe` | WebXR / 3D | HTML + JS | A-Frame |
| `threejs` | 3D graphics | HTML + JS | Three.js |

Anything outside these six domains is rejected at the policy layer.

---

## Supported Models

RNNR_Coding supports **only** two models:

| Model | Min RAM | Use Case |
|-------|---------|----------|
| `qwen3:8b` | 8 GB | Lightweight, general coding tasks |
| `qwen3-coder:30b` | 32 GB | Full capacity, complex generation |

At startup, if `--model auto` is used (default), the system detects available RAM and selects the appropriate model automatically.

---

## Installation

```bash
# 1. Install Ollama
# See https://ollama.ai for platform-specific instructions

# 2. Pull a supported model
ollama pull qwen3:8b
# or
ollama pull qwen3-coder:30b

# 3. Install RNNR_Coding
git clone https://github.com/your-org/RNNR_Coding.git
cd RNNR_Coding
pip install -e .
```

---

## Usage

### Basic Usage

```bash
# Start in Python fundamentals mode
rnnr --mode python_basic

# Start in pandas mode with research logging
rnnr --mode pandas --research

# Start in A-Frame mode with debug output
rnnr --mode aframe --debug

# Use a specific model
rnnr --mode sklearn --model qwen3-coder:30b
```

### Interactive Commands

| Command | Description |
|---------|-------------|
| `/help` | Show available commands |
| `/exit` | Exit the session |
| `/clear` | Clear conversation history |
| `/metrics` | Show session metrics |
| `/trace` | Show last LLM interaction |
| `/status` | Show session status |
| `/save` | Save session to disk |

### CLI Flags

| Flag | Description |
|------|-------------|
| `--mode` | **Required.** One of: `python_basic`, `py5`, `sklearn`, `pandas`, `aframe`, `threejs` |
| `--model` | Model name or `auto` (default: `auto`) |
| `--safe-mode` | Enable extra safety restrictions |
| `--debug` | Print debug information |
| `--max-iterations` | Max agent loop iterations (default: 50) |
| `--research` | Enable research mode logging |
| `--resume SESSION_ID` | Resume a previous session |
| `--list-sessions` | List all saved sessions |
| `--export-session SESSION_ID` | Export session to JSONL |

---

## Architecture

```
rnnr/
├── main.py              # Entry point, interactive loop
├── cli.py               # Argument parser
├── agent_loop.py        # Core agentic loop (<200 lines)
├── llm_adapter.py       # Ollama API communication
├── tool_registry.py     # Tool instantiation and dispatch
├── security_manager.py  # Command and path blocklists
├── policy_engine.py     # Mode-specific policy enforcement
├── validator.py         # Language isolation and output validation
├── session_manager.py   # Session persistence and export
├── metrics.py           # Iteration and usage tracking
├── config.py            # Models, modes, constants
└── tools/
    ├── bash_tool.py     # Shell command execution
    ├── read_tool.py     # File reading
    ├── write_tool.py    # File writing
    ├── edit_tool.py     # String replacement editing
    ├── grep_tool.py     # Regex content search
    └── glob_tool.py     # File pattern matching
```

### Mode System

Each mode defines a `ModeProfile` that specifies:

- **System prompt** — Instructs the LLM to stay within the domain
- **Language family** — Python or Web (HTML+JS)
- **Allowed imports** — Whitelist of permitted Python modules
- **Allowed file extensions** — Only `.py` in Python modes, only `.html/.js/.css` in web modes
- **Allowed tools** — Web modes cannot use `bash` to prevent arbitrary execution

The mode is selected at startup via `--mode` and cannot be changed during a session.

### Validation Layer

The `Validator` inspects every piece of generated code before it reaches the user:

1. **Language isolation** — Detects HTML/JS in Python mode and Python in web mode using heuristic pattern matching
2. **Import whitelist** — Parses Python AST to extract imports, rejects anything not in the mode's allow-list
3. **Line count limit** — Rejects output exceeding 300 lines (configurable)
4. **File count limit** — Rejects multi-file output (default: 1 file per response)
5. **Forbidden patterns** — Blocks URLs, `pip install`, `os.system()`, `subprocess` usage
6. **Retry mechanism** — On violation, generates a correction prompt and re-queries the LLM

When a violation is detected:
- The response is rejected (never shown to the user)
- A correction prompt is injected into the conversation
- The LLM is re-queried with explicit instructions to fix the issue
- The violation is logged (in research mode)

### Security Model

RNNR_Coding enforces a strict security perimeter:

**Blocked at shell level:**
- `sudo`, `chmod`, `chown`, `dd`, `mkfs`, `rm -rf /`
- `curl`, `wget`, `ssh`, `scp`, `nc`, `nmap`
- `pip install`, `npm install`, `yarn add`
- `docker`, `systemctl`, `kill`, `shutdown`, `reboot`
- `curl | bash` piping patterns

**Blocked at file level:**
- Writing outside the working directory
- Symlink traversal
- File extension violations

**Blocked at code level:**
- Network libraries (`requests`, `httpx`, `socket`, `urllib`)
- System libraries (`subprocess`, `os.system`, `shutil`, `ctypes`)
- URL references in generated code

This system is designed to be **strictly offline and curriculum-limited**.

---

## Research Mode

When started with `--research`, RNNR_Coding logs every interaction in structured JSONL format:

```bash
rnnr --mode python_basic --research
```

Each session produces a reproducible log containing:

- All user prompts
- All LLM outputs (raw)
- All validation failures with rule and detail
- All language violation detections
- All tool calls and results
- Iteration counts
- Selected mode and model
- Timing information

### Exporting Research Data

```bash
# List all sessions
rnnr --mode python_basic --list-sessions

# Export a specific session
rnnr --mode python_basic --export-session abc123def456
```

The JSONL format enables direct analysis with `pandas`, `jq`, or any log-processing pipeline.

---

## Comparison with Generic Coding Agents

| Feature | Generic Agent | RNNR_Coding |
|---------|--------------|-------------|
| Language scope | Any | 6 fixed domains |
| Model support | Cloud APIs | Local only (Qwen3) |
| Language mixing | Unrestricted | Strictly prohibited |
| Import control | None | Whitelist per mode |
| Network access | Allowed | Blocked |
| Package installation | Allowed | Blocked |
| Research logging | Varies | Structured JSONL |
| Session reproducibility | No | Yes |
| Output validation | None | Multi-layer |

---

## Educational Usage Example

**Scenario:** A university course on data analysis using pandas.

1. Instructor configures student environments with Ollama + `qwen3:8b`
2. Students launch: `rnnr --mode pandas --research`
3. Students interact with the agent to learn pandas operations
4. The agent **cannot** generate Flask web apps, React components, or Rust code — even if asked
5. All sessions are logged for the instructor to review learning patterns
6. The instructor exports JSONL logs for research on AI-assisted pedagogy

---

## Roadmap

- [ ] **v0.2** — Curriculum progression tracking (beginner → intermediate → advanced)
- [ ] **v0.3** — Multi-file project mode with configurable file limits
- [ ] **v0.4** — Built-in exercise/challenge system with auto-grading
- [ ] **v0.5** — Instructor dashboard for session analytics
- [ ] **v0.6** — Support for additional Qwen model variants as they release
- [ ] **v0.7** — Jupyter notebook integration for pandas/sklearn modes
- [ ] **v0.8** — Live preview server for A-Frame/Three.js modes (localhost only)
- [ ] **v1.0** — Stable release with full documentation and deployment guides

---

## License

MIT License. See [LICENSE](LICENSE) for details.

---

## Contributing

Contributions are welcome. Please ensure all changes:

1. Maintain strict mode isolation
2. Do not expand the supported model list without explicit discussion
3. Include tests for new validation rules
4. Follow the existing code style (type hints, docstrings, no external dependencies beyond Ollama)
