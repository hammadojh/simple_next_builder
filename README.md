# Simple Next Builder

An AI-powered tool that automates NextJS app creation from simple prompts.

## Setup

1. **Install dependencies**:
   ```bash
   pip install -r requirements_web.txt
   ```

2. **Set up API key**:
   - Create a `.env` file with your OpenAI API key:
     ```
     OPENAI_API_KEY=your_key_here
     ```
   - Or get an OpenRouter key: `OPENROUTER_API_KEY=your_key_here`
   - Get keys from: [OpenAI](https://platform.openai.com/api-keys) or [OpenRouter](https://openrouter.ai/)

## Running

### Web Interface (Recommended)
```bash
python src/webapp/start_web.py
```
Then open: http://localhost:8001

### Command Line
```bash
# Create a new app
python src/main.py --idea "A todo list app with dark mode"

# Edit existing app
python src/main.py --edit myapp1 --idea "Add a reset button"

# Interactive mode
python src/main.py --interactive

# List existing apps
python src/main.py --list
```

## ✨ NEW: LLM Coordinator System

The latest version includes an **intelligent LLM coordinator** that acts as a "senior developer" to plan and orchestrate complex app creation and editing tasks.

### 🧠 How the Coordinator Works

Instead of sending prompts directly to code generation LLMs, the coordinator:

1. **📋 Analyzes** the user request and creates a detailed execution plan
2. **🔧 Breaks down** complex requests into smaller, manageable tasks  
3. **🎯 Decides** what context and information each task needs
4. **🚀 Coordinates** multiple strategic LLM calls
5. **✅ Tests** and validates results at each step
6. **🛡️ Handles** failures with automatic recovery strategies

### 🎯 Benefits

- **Better Complex Edits**: Can handle multi-step changes across multiple files
- **Smarter Context**: Only gathers relevant information for each task
- **More Reliable**: Validates each step before proceeding  
- **Self-Healing**: Automatically recovers from failures
- **Strategic Planning**: Optimizes task order and dependencies

### 🔧 Usage

The coordinator is **enabled by default**. You can control it with:

```bash
# Use coordinator (default behavior)
python src/main.py --idea "Complex e-commerce site with user auth"

# Disable coordinator for simple tasks
python src/main.py --idea "Simple counter" --no-coordinator

# Explicitly enable coordinator  
python src/main.py --idea "Multi-page blog platform" --coordinator
```

### 📊 When to Use Each Mode

**Coordinator Mode (Recommended)**:
- ✅ Complex multi-feature applications
- ✅ Major edits affecting multiple files
- ✅ Apps requiring architectural planning
- ✅ When you want maximum reliability

**Traditional Mode**:
- ✅ Simple single-page apps
- ✅ Quick prototypes
- ✅ Minor edits to existing apps
- ✅ When you prefer direct LLM interaction

## How it works

### Traditional Mode
1. Takes your app idea prompt
2. Creates a NextJS template
3. Uses AI to generate custom code
4. Applies changes automatically 
5. Installs dependencies and runs the app

### Coordinator Mode (NEW)
1. **Planning Phase**: Coordinator LLM analyzes request and creates execution plan
2. **Task Breakdown**: Complex request split into focused subtasks
3. **Context Gathering**: Each task gets only the relevant context it needs
4. **Coordinated Execution**: Multiple specialized LLM calls execute the plan
5. **Validation & Recovery**: Each step is validated with automatic error recovery
6. **Build & Deploy**: Final validation and app launch

Apps are created in the `apps/` directory. 