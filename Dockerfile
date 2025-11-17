# Image de base
FROM python:3.11-slim

# Répertoire de travail
WORKDIR /app

# Copier requirements et installer les dépendances
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copier les fichiers nécessaires
COPY app.py .
COPY train_model.py .
COPY models/ ./models/

# Créer le dossier plots
RUN mkdir -p plots

# Exposer le port
EXPOSE 7860

# Variables d'environnement Gradio
ENV GRADIO_SERVER_NAME=0.0.0.0
ENV GRADIO_SERVER_PORT=7860

# Commande de démarrage
CMD ["python", "app.py"]