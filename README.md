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

## How it works

1. Takes your app idea prompt
2. Creates a NextJS template
3. Uses AI to generate custom code
4. Applies changes automatically 
5. Installs dependencies and runs the app

Apps are created in the `apps/` directory. 