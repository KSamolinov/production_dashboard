# Используем официальный образ Python
FROM python:3.12-slim

# Устанавливаем зависимости системы
RUN apt-get update && apt-get install -y \
    libjpeg-dev zlib1g-dev gcc \
    && rm -rf /var/lib/apt/lists/*

# Задаём рабочую директорию
WORKDIR /app

# Копируем файлы проекта
COPY . .

# Устанавливаем Python-зависимости
RUN pip install --no-cache-dir -r requirements.txt

# Указываем порт (Flask по умолчанию 5000)
EXPOSE 5000

# Переменная окружения для Flask
ENV FLASK_APP=app/main.py
ENV FLASK_RUN_HOST=0.0.0.0
ENV FLASK_ENV=production

# Запуск приложения
CMD ["flask", "run"]
