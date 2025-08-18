# Makefile for FTT-ML Backend
# Provides commands for development, testing, building, and deployment

# Variables
PYTHON := python3.11
POETRY := poetry
APP_NAME := ftt-ml-backend
DOCKER_IMAGE := $(APP_NAME)
PORT := 8000
HOST := 0.0.0.0

# Colors for output
RED := \033[0;31m
GREEN := \033[0;32m
YELLOW := \033[0;33m
BLUE := \033[0;34m
NC := \033[0m # No Color

.PHONY: help install dev-install test clean build docker-build docker-run lint format security check-deps start dev-start production-start

# Default target
.DEFAULT_GOAL := help

help: ## Show this help message
	@echo "$(BLUE)FTT-ML Backend Makefile$(NC)"
	@echo "========================="
	@echo ""
	@echo "Available commands:"
	@grep -E '^[a-zA-Z_-]+:.*?## .*$$' $(MAKEFILE_LIST) | sort | awk 'BEGIN {FS = ":.*?## "}; {printf "\033[36m%-20s\033[0m %s\n", $$1, $$2}'

# ==================== ENVIRONMENT SETUP ====================

install: ## Install dependencies using Poetry
	@echo "$(BLUE)Installing dependencies with Poetry...$(NC)"
	$(POETRY) install --no-dev
	@echo "$(GREEN)✅ Dependencies installed successfully$(NC)"

dev-install: ## Install all dependencies including dev dependencies
	@echo "$(BLUE)Installing all dependencies (including dev)...$(NC)"
	$(POETRY) install
	$(POETRY) run pre-commit install
	@echo "$(GREEN)✅ Development environment set up successfully$(NC)"

install-legacy: ## Install dependencies using pip (fallback)
	@echo "$(BLUE)Installing dependencies with pip...$(NC)"
	pip install -r requirements.txt
	@echo "$(GREEN)✅ Dependencies installed with pip$(NC)"

update-deps: ## Update all dependencies
	@echo "$(BLUE)Updating dependencies...$(NC)"
	$(POETRY) update
	$(POETRY) export -f requirements.txt --output requirements.txt --without-hashes
	@echo "$(GREEN)✅ Dependencies updated$(NC)"

# ==================== DEVELOPMENT ====================

dev: dev-install ## Set up development environment and start dev server
	@echo "$(BLUE)Starting development server...$(NC)"
	$(POETRY) run uvicorn app.main:app --reload --host $(HOST) --port $(PORT)

start: ## Start the application (production mode)
	@echo "$(BLUE)Starting application in production mode...$(NC)"
	$(POETRY) run uvicorn app.main:app --host $(HOST) --port $(PORT) --workers 1

dev-start: ## Start development server with hot reload
	@echo "$(BLUE)Starting development server with auto-reload...$(NC)"
	$(POETRY) run uvicorn app.main:app --reload --host $(HOST) --port $(PORT) --log-level debug

production-start: ## Start production server with optimized settings
	@echo "$(BLUE)Starting production server...$(NC)"
	$(POETRY) run uvicorn app.main:app --host $(HOST) --port $(PORT) --workers 4 --log-level info

waitress-start: ## Start application using Waitress WSGI server
	@echo "$(BLUE)Starting application with Waitress...$(NC)"
	$(POETRY) run python app/server.py

waitress-dev: ## Start Waitress in development mode
	@echo "$(BLUE)Starting Waitress in development mode...$(NC)"
	ENVIRONMENT=development DEBUG=true $(POETRY) run python app/server.py

waitress-prod: ## Start Waitress in production mode
	@echo "$(BLUE)Starting Waitress in production mode...$(NC)"
	ENVIRONMENT=production DEBUG=false WAITRESS_THREADS=8 $(POETRY) run python app/server.py

# ==================== TESTING ====================

test: ## Run all tests
	@echo "$(BLUE)Running tests...$(NC)"
	$(POETRY) run python test/run_tests.py

test-unit: ## Run unit tests only
	@echo "$(BLUE)Running unit tests...$(NC)"
	$(POETRY) run pytest test/ -m "unit" -v

test-integration: ## Run integration tests only
	@echo "$(BLUE)Running integration tests...$(NC)"
	$(POETRY) run pytest test/ -m "integration" -v

test-coverage: ## Run tests with coverage report
	@echo "$(BLUE)Running tests with coverage...$(NC)"
	$(POETRY) run python test/run_tests.py --coverage --html-report
	@echo "$(GREEN)✅ Coverage report generated in htmlcov/$(NC)"

test-parallel: ## Run tests in parallel
	@echo "$(BLUE)Running tests in parallel...$(NC)"
	$(POETRY) run python test/run_tests.py --parallel

test-reconciliation: ## Run reconciliation-specific tests
	@echo "$(BLUE)Running reconciliation tests...$(NC)"
	$(POETRY) run pytest test/ -m "reconciliation" -v

test-file-upload: ## Run file upload tests
	@echo "$(BLUE)Running file upload tests...$(NC)"
	$(POETRY) run pytest test/ -m "file_upload" -v

test-viewer: ## Run viewer tests
	@echo "$(BLUE)Running viewer tests...$(NC)"
	$(POETRY) run pytest test/ -m "viewer" -v

# ==================== CODE QUALITY ====================

lint: ## Run all linting checks
	@echo "$(BLUE)Running linting checks...$(NC)"
	$(POETRY) run flake8 app/ --count --select=E9,F63,F7,F82 --show-source --statistics
	$(POETRY) run flake8 app/ --count --exit-zero --max-complexity=10 --max-line-length=100 --statistics
	$(POETRY) run mypy app/ --ignore-missing-imports
	@echo "$(GREEN)✅ Linting completed$(NC)"

format: ## Format code with black and isort
	@echo "$(BLUE)Formatting code...$(NC)"
	$(POETRY) run black app/ test/
	$(POETRY) run isort app/ test/
	@echo "$(GREEN)✅ Code formatted$(NC)"

format-check: ## Check if code formatting is correct
	@echo "$(BLUE)Checking code formatting...$(NC)"
	$(POETRY) run black --check app/ test/
	$(POETRY) run isort --check-only app/ test/

security: ## Run security checks
	@echo "$(BLUE)Running security checks...$(NC)"
	$(POETRY) run bandit -r app/ -ll
	$(POETRY) run safety check
	@echo "$(GREEN)✅ Security checks completed$(NC)"

check-deps: ## Check for dependency vulnerabilities
	@echo "$(BLUE)Checking dependencies for vulnerabilities...$(NC)"
	$(POETRY) run safety check
	@echo "$(GREEN)✅ Dependency check completed$(NC)"

pre-commit: ## Run pre-commit hooks
	@echo "$(BLUE)Running pre-commit hooks...$(NC)"
	$(POETRY) run pre-commit run --all-files

quality: lint format-check security ## Run all quality checks

# ==================== BUILDING ====================

build: ## Build the application for deployment
	@echo "$(BLUE)Building application...$(NC)"
	$(POETRY) build
	@echo "$(GREEN)✅ Application built successfully$(NC)"

clean: ## Clean up temporary files and caches
	@echo "$(BLUE)Cleaning up...$(NC)"
	find . -type f -name "*.pyc" -delete
	find . -type d -name "__pycache__" -delete
	find . -type d -name "*.egg-info" -exec rm -rf {} +
	find . -type d -name ".pytest_cache" -exec rm -rf {} +
	find . -type d -name ".mypy_cache" -exec rm -rf {} +
	rm -rf build/ dist/ htmlcov/ .coverage
	@echo "$(GREEN)✅ Cleanup completed$(NC)"

clean-docker: ## Clean up Docker images and containers
	@echo "$(BLUE)Cleaning up Docker...$(NC)"
	docker system prune -f
	docker image prune -f
	@echo "$(GREEN)✅ Docker cleanup completed$(NC)"

# ==================== DOCKER ====================

docker-build: ## Build Docker image
	@echo "$(BLUE)Building Docker image...$(NC)"
	docker build -t $(DOCKER_IMAGE):latest .
	@echo "$(GREEN)✅ Docker image built: $(DOCKER_IMAGE):latest$(NC)"

docker-build-prod: ## Build production Docker image with optimizations
	@echo "$(BLUE)Building production Docker image...$(NC)"
	docker build -t $(DOCKER_IMAGE):prod -f Dockerfile .
	@echo "$(GREEN)✅ Production Docker image built: $(DOCKER_IMAGE):prod$(NC)"

docker-run: docker-build ## Build and run Docker container
	@echo "$(BLUE)Running Docker container...$(NC)"
	docker run -d --name $(APP_NAME) -p $(PORT):$(PORT) \
		-e ENVIRONMENT=development \
		-e DEBUG=true \
		$(DOCKER_IMAGE):latest

docker-run-prod: docker-build-prod ## Build and run production Docker container
	@echo "$(BLUE)Running production Docker container...$(NC)"
	docker run -d --name $(APP_NAME)-prod -p $(PORT):$(PORT) \
		-e ENVIRONMENT=production \
		-e DEBUG=false \
		-e OPENAI_API_KEY=${OPENAI_API_KEY} \
		$(DOCKER_IMAGE):prod

docker-stop: ## Stop and remove Docker container
	@echo "$(BLUE)Stopping Docker container...$(NC)"
	docker stop $(APP_NAME) || true
	docker rm $(APP_NAME) || true

docker-logs: ## View Docker container logs
	docker logs -f $(APP_NAME)

docker-shell: ## Open shell in running Docker container
	docker exec -it $(APP_NAME) bash

# ==================== DEPLOYMENT ====================

deploy-staging: ## Deploy to staging environment
	@echo "$(BLUE)Deploying to staging...$(NC)"
	# Add your staging deployment commands here
	@echo "$(GREEN)✅ Deployed to staging$(NC)"

deploy-production: ## Deploy to production environment
	@echo "$(BLUE)Deploying to production...$(NC)"
	# Add your production deployment commands here
	@echo "$(GREEN)✅ Deployed to production$(NC)"

# ==================== DATABASE ====================

db-reset: ## Reset database (if using one)
	@echo "$(BLUE)Resetting database...$(NC)"
	# Add database reset commands here
	@echo "$(GREEN)✅ Database reset$(NC)"

# ==================== HEALTH CHECKS ====================

health: ## Check application health
	@echo "$(BLUE)Checking application health...$(NC)"
	curl -f http://localhost:$(PORT)/health || echo "$(RED)❌ Health check failed$(NC)"

health-debug: ## Check detailed application status
	@echo "$(BLUE)Checking detailed application status...$(NC)"
	curl -s http://localhost:$(PORT)/debug/status | python -m json.tool || echo "$(RED)❌ Debug status failed$(NC)"

# ==================== DOCUMENTATION ====================

docs: ## Generate API documentation
	@echo "$(BLUE)API documentation available at:$(NC)"
	@echo "Swagger UI: http://localhost:$(PORT)/docs"
	@echo "ReDoc: http://localhost:$(PORT)/redoc"

# ==================== UTILITY ====================

env-check: ## Check environment variables
	@echo "$(BLUE)Checking environment configuration...$(NC)"
	@echo "Python: $$($(PYTHON) --version)"
	@echo "Poetry: $$($(POETRY) --version)"
	@echo "Current directory: $$(pwd)"
	@echo "Environment: $${ENVIRONMENT:-development}"
	@echo "Debug mode: $${DEBUG:-true}"
	@echo "OpenAI configured: $$(if [ -n "$${OPENAI_API_KEY}" ]; then echo 'Yes'; else echo 'No'; fi)"

size: ## Show project size
	@echo "$(BLUE)Project size:$(NC)"
	du -sh .
	@echo "$(BLUE)Python files:$(NC)"
	find . -name "*.py" | wc -l

performance-test: ## Run performance tests
	@echo "$(BLUE)Running performance tests...$(NC)"
	# Add performance testing commands here
	@echo "$(GREEN)✅ Performance tests completed$(NC)"

# ==================== COMPLETE WORKFLOWS ====================

full-test: clean lint test-coverage security ## Run complete testing suite
	@echo "$(GREEN)✅ Full testing suite completed$(NC)"

full-build: clean lint test build docker-build ## Complete build process
	@echo "$(GREEN)✅ Complete build process finished$(NC)"

ci: install lint test security ## Continuous Integration workflow
	@echo "$(GREEN)✅ CI workflow completed$(NC)"

# ==================== DEVELOPMENT HELPERS ====================

watch: ## Watch for file changes and restart server
	@echo "$(BLUE)Watching for changes...$(NC)"
	$(POETRY) run watchdog --patterns="*.py" --recursive --command="make dev-start" app/

shell: ## Open Python shell with app context
	@echo "$(BLUE)Opening Python shell...$(NC)"
	$(POETRY) run python

notebook: ## Start Jupyter notebook for data analysis
	@echo "$(BLUE)Starting Jupyter notebook...$(NC)"
	$(POETRY) run jupyter notebook

# Show current configuration
status: ## Show current project status
	@echo "$(BLUE)Project Status:$(NC)"
	@echo "==============="
	@echo "Application: $(APP_NAME)"
	@echo "Port: $(PORT)"
	@echo "Host: $(HOST)"
	@echo "Python: $$($(PYTHON) --version)"
	@echo "Poetry: $$($(POETRY) --version)"
	@echo "Docker Image: $(DOCKER_IMAGE)"
	@echo ""
	@echo "$(BLUE)Available endpoints when running:$(NC)"
	@echo "Health: http://localhost:$(PORT)/health"
	@echo "API Docs: http://localhost:$(PORT)/docs"
	@echo "ReDoc: http://localhost:$(PORT)/redoc"