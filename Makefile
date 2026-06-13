.PHONY: setup download train tune backend frontend

setup:
	pip install -r requirements.txt
	cd frontend && npm install

download:
	python src/data/download.py

train:
	python src/training/train.py

tune:
	python src/training/optuna_tune.py

backend:
	python backend/main.py

frontend:
	cd frontend && npm run dev
