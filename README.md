# Satellite-based Deforestation and Land Cover classification
## How to install
1. Agree to the rules of [this Kaggle competition](https://www.kaggle.com/competitions/planet-understanding-the-amazon-from-space/data)
2. Make sure you have a Kaggle API key and set in either in `.env` or `~/.kaggle`. Kaggle should prompt you to do this upon key creation
3. Install uv and make sure it's in your `PATH`
4. run `uv sync` to sync your virtual environment

## Development tips
1. Use `uv run main.py` to run the script
2. If you need to install a package that has not yet been added, use `uv add [package]`. Projects with uv do not require pip
3. If somebody else adds a package, run `uv sync` to make sure your virtual environment is up to date
