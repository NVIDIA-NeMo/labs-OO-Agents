#!/bin/bash
set -e

echo "🚀 Setting up nooa development environment..."

# Check if uv is installed
if ! command -v uv &> /dev/null; then
    echo "❌ Error: uv not found. Please install uv first:"
    echo "   curl -LsSf https://astral.sh/uv/install.sh | sh"
    echo "   Or visit: https://docs.astral.sh/uv/getting-started/installation/"
    exit 1
fi

echo "✓ Found uv $(uv --version)"

# Sync dependencies with uv (creates venv and installs Python if needed)
echo "📦 Syncing dependencies with uv..."
uv sync --all-extras

uv run pre-commit install

echo ""
echo "✅ Setup complete!"
echo ""
echo "Next steps:"
echo "  1. Activate the environment:"
echo "     source .venv/bin/activate"
echo ""
echo "  2. Set your API key (create .env in repo root if it doesn't exist):"
echo "     [ ! -f .env ] && echo 'NVIDIA_INFERENCE_API_KEY=your-key-here' > .env"
echo ""
echo "  3. Run tests:"
echo "     uv run pytest tests/"
echo ""
echo "  4. Try a quickstart example:"
echo "     uv run python examples/quickstart/01_first_generation_method.py"
echo ""
echo "💡 Tip: You can run commands without activating venv using 'uv run'"
echo ""
