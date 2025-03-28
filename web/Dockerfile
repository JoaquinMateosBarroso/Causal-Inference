# 
FROM python:3.9

# 
WORKDIR /app



# Update package list and install Graphviz if it's not already installed
RUN apt-get update && \
    apt-get install -y graphviz

# Install necessary Python packages
COPY ./requirements.txt /app/requirements.txt 
RUN pip install --no-cache-dir --upgrade -r /app/requirements.txt

# 
COPY ./app /app

# 
CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000"]