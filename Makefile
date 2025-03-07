# Makefile for Custom Transformer AI

# Define variables
PORT=8000
PYTHON=python

.DEFAULT_GOAL := explain
explain:
	### Welcome
	### Targets
	@echo " Choose a command to run: "
	@cat Makefile* | grep -E '^[a-zA-Z_-]+:.*?## .*$$' | sort | awk 'BEGIN {FS = ":.*?## "}; {printf "\033[36m%-30s\033[0m %s\n", $$1, $$2}'

install: ## Install dependencies
	pip install -r requirements.txt

train-tokenizer: ## Train the tokenizer
	$(PYTHON) scripts/train_tokenizer.py

train-model: ## Train the transformer model
	$(PYTHON) scripts/train_transformer.py

start-api: free-port ## Start FastAPI server
	uvicorn api.server:app --reload

test-api: ## Test API
	curl "http://127.0.0.1:8000/generate?prompt=What%20is%20Python?"

start-ui: ## Start the Streamlit UI
	streamlit run app.py

run-all: install train-tokenizer train-model start-api ## Run everything in order

free-port: ## Kill process using port 8000
	@PID=$$(lsof -ti :$(PORT)); \
	if [ "$$PID" ]; then \
		echo "Killing process on port $(PORT) (PID: $$PID)..."; \
		kill -9 $$PID; \
		echo "Port $(PORT) is now free."; \
	else \
		echo "No process found on port $(PORT)."; \
	fi
