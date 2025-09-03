# RunPod Jupyter Template - Based on vLLM OpenAI Image
FROM vllm/vllm-openai:v0.10.1.1
# Set environment variables
ENV DEBIAN_FRONTEND=noninteractive
ENV PYTHONUNBUFFERED=1
ENV JUPYTER_ENABLE_LAB=yes
# Override vLLM usage source
ENV VLLM_USAGE_SOURCE=runpod-jupyter-template

# Set working directory
WORKDIR /

# Update system and install additional dependencies
RUN apt-get update && apt-get install -y \
    git \
    vim \
    nano \
    htop \
    nvtop \
    curl \
    wget \
    zip \
    unzip \
    openssh-server \
    nginx \
    && rm -rf /var/lib/apt/lists/*

# Fix package conflicts - handle distutils packages differently
RUN pip uninstall -y fsspec dill || true

# For blinker (distutils package), we need to work around it
RUN pip install --no-cache-dir --force-reinstall --break-system-packages \
    fsspec \
    dill

# Install blinker with ignore-installed to avoid conflict
RUN pip install --no-cache-dir --ignore-installed blinker

# Install Jupyter core packages first
RUN pip install --no-cache-dir \
    jupyterlab==4.0.7 \
    jupyter==1.0.0 \
    notebook==7.0.6

# Install Jupyter widgets and extensions
RUN pip install --no-cache-dir \
    ipywidgets==8.1.1 \
    jupyterlab-widgets

# Install data science packages
RUN pip install --no-cache-dir \
    matplotlib \
    seaborn \
    pandas \
    numpy \
    scikit-learn

# Install visualization packages
RUN pip install --no-cache-dir \
    plotly \
    dash

# Install image processing
RUN pip install --no-cache-dir \
    opencv-python-headless \
    Pillow

# Install ML/AI packages
RUN pip install --no-cache-dir \
    datasets \
    accelerate \
    wandb \
    tensorboard

# Install web frameworks
RUN pip install --no-cache-dir \
    gradio \
    streamlit \
    fastapi \
    uvicorn

# Install JupyterLab extensions
RUN pip install --no-cache-dir \
    jupyterlab-git \
    jupyterlab-lsp \
    python-lsp-server \
    jupyterlab_code_formatter \
    black \
    isort

# Create workspace directory
RUN mkdir -p /workspace

# Configure SSH
RUN echo "PermitRootLogin yes" >> /etc/ssh/sshd_config && \
    echo "PasswordAuthentication yes" >> /etc/ssh/sshd_config && \
    echo "PubkeyAuthentication yes" >> /etc/ssh/sshd_config

# Configure Nginx
RUN echo 'server {' > /etc/nginx/sites-available/default && \
    echo '    listen 80;' >> /etc/nginx/sites-available/default && \
    echo '    server_name _;' >> /etc/nginx/sites-available/default && \
    echo '    location / {' >> /etc/nginx/sites-available/default && \
    echo '        proxy_pass http://127.0.0.1:8888;' >> /etc/nginx/sites-available/default && \
    echo '        proxy_set_header Host $host;' >> /etc/nginx/sites-available/default && \
    echo '        proxy_set_header X-Real-IP $remote_addr;' >> /etc/nginx/sites-available/default && \
    echo '        proxy_set_header X-Forwarded-For $proxy_add_x_forwarded_for;' >> /etc/nginx/sites-available/default && \
    echo '        proxy_set_header X-Forwarded-Proto $scheme;' >> /etc/nginx/sites-available/default && \
    echo '        proxy_http_version 1.1;' >> /etc/nginx/sites-available/default && \
    echo '        proxy_set_header Upgrade $http_upgrade;' >> /etc/nginx/sites-available/default && \
    echo '        proxy_set_header Connection "upgrade";' >> /etc/nginx/sites-available/default && \
    echo '    }' >> /etc/nginx/sites-available/default && \
    echo '}' >> /etc/nginx/sites-available/default

# Create the RunPod start script with vLLM integration
RUN echo '#!/bin/bash' > /start.sh && \
    echo 'set -e  # Exit the script if any statement returns a non-true return value' >> /start.sh && \
    echo '' >> /start.sh && \
    echo '# ---------------------------------------------------------------------------- #' >> /start.sh && \
    echo '#                          Function Definitions                                #' >> /start.sh && \
    echo '# ---------------------------------------------------------------------------- #' >> /start.sh && \
    echo '' >> /start.sh && \
    echo '# Start nginx service' >> /start.sh && \
    echo 'start_nginx() {' >> /start.sh && \
    echo '    echo "Starting Nginx service..."' >> /start.sh && \
    echo '    service nginx start' >> /start.sh && \
    echo '}' >> /start.sh && \
    echo '' >> /start.sh && \
    echo '# Execute script if exists' >> /start.sh && \
    echo 'execute_script() {' >> /start.sh && \
    echo '    local script_path=$1' >> /start.sh && \
    echo '    local script_msg=$2' >> /start.sh && \
    echo '    if [[ -f ${script_path} ]]; then' >> /start.sh && \
    echo '        echo "${script_msg}"' >> /start.sh && \
    echo '        bash ${script_path}' >> /start.sh && \
    echo '    fi' >> /start.sh && \
    echo '}' >> /start.sh && \
    echo '' >> /start.sh && \
    echo '# Setup ssh' >> /start.sh && \
    echo 'setup_ssh() {' >> /start.sh && \
    echo '    if [[ $PUBLIC_KEY ]]; then' >> /start.sh && \
    echo '        echo "Setting up SSH..."' >> /start.sh && \
    echo '        mkdir -p ~/.ssh' >> /start.sh && \
    echo '        echo "$PUBLIC_KEY" >> ~/.ssh/authorized_keys' >> /start.sh && \
    echo '        chmod 700 -R ~/.ssh' >> /start.sh && \
    echo '         if [ ! -f /etc/ssh/ssh_host_rsa_key ]; then' >> /start.sh && \
    echo '            ssh-keygen -t rsa -f /etc/ssh/ssh_host_rsa_key -q -N '\'''\''' >> /start.sh && \
    echo '            echo "RSA key fingerprint:"' >> /start.sh && \
    echo '            ssh-keygen -lf /etc/ssh/ssh_host_rsa_key.pub' >> /start.sh && \
    echo '        fi' >> /start.sh && \
    echo '        if [ ! -f /etc/ssh/ssh_host_dsa_key ]; then' >> /start.sh && \
    echo '            ssh-keygen -t dsa -f /etc/ssh/ssh_host_dsa_key -q -N '\'''\''' >> /start.sh && \
    echo '            echo "DSA key fingerprint:"' >> /start.sh && \
    echo '            ssh-keygen -lf /etc/ssh/ssh_host_dsa_key.pub' >> /start.sh && \
    echo '        fi' >> /start.sh && \
    echo '        if [ ! -f /etc/ssh/ssh_host_ecdsa_key ]; then' >> /start.sh && \
    echo '            ssh-keygen -t ecdsa -f /etc/ssh/ssh_host_ecdsa_key -q -N '\'''\''' >> /start.sh && \
    echo '            echo "ECDSA key fingerprint:"' >> /start.sh && \
    echo '            ssh-keygen -lf /etc/ssh/ssh_host_ecdsa_key.pub' >> /start.sh && \
    echo '        fi' >> /start.sh && \
    echo '        if [ ! -f /etc/ssh/ssh_host_ed25519_key ]; then' >> /start.sh && \
    echo '            ssh-keygen -t ed25519 -f /etc/ssh/ssh_host_ed25519_key -q -N '\'''\''' >> /start.sh && \
    echo '            echo "ED25519 key fingerprint:"' >> /start.sh && \
    echo '            ssh-keygen -lf /etc/ssh/ssh_host_ed25519_key.pub' >> /start.sh && \
    echo '        fi' >> /start.sh && \
    echo '        service ssh start' >> /start.sh && \
    echo '        echo "SSH host keys:"' >> /start.sh && \
    echo '        for key in /etc/ssh/*.pub; do' >> /start.sh && \
    echo '            echo "Key: $key"' >> /start.sh && \
    echo '            ssh-keygen -lf $key' >> /start.sh && \
    echo '        done' >> /start.sh && \
    echo '    fi' >> /start.sh && \
    echo '}' >> /start.sh && \
    echo '' >> /start.sh && \
    echo '# Export env vars' >> /start.sh && \
    echo 'export_env_vars() {' >> /start.sh && \
    echo '    echo "Exporting environment variables..."' >> /start.sh && \
    echo '    printenv | grep -E '\''^RUNPOD_|^PATH=|^_=|^VLLM_'\'' | awk -F = '\''{ print "export " $1 "=\"" $2 "\"" }'\'' >> /etc/rp_environment' >> /start.sh && \
    echo '    echo '\''source /etc/rp_environment'\'' >> ~/.bashrc' >> /start.sh && \
    echo '}' >> /start.sh && \
    echo '' >> /start.sh && \
    echo '# Start jupyter lab' >> /start.sh && \
    echo 'start_jupyter() {' >> /start.sh && \
    echo '    echo "Starting Jupyter Lab..."' >> /start.sh && \
    echo '    mkdir -p /workspace && \' >> /start.sh && \
    echo '    cd / && \' >> /start.sh && \
    echo '    # Set default token if not provided' >> /start.sh && \
    echo '    JUPYTER_TOKEN=${JUPYTER_PASSWORD:-"runpod"}' >> /start.sh && \
    echo '    nohup python3 -m jupyter lab --allow-root --no-browser --port=8888 --ip=* --FileContentsManager.delete_to_trash=False --ServerApp.terminado_settings='\''{"shell_command":["/bin/bash"]}'\'' --ServerApp.token=$JUPYTER_TOKEN --ServerApp.allow_origin=* --ServerApp.preferred_dir=/workspace &> /jupyter.log &' >> /start.sh && \
    echo '    echo "Jupyter Lab started with token: $JUPYTER_TOKEN"' >> /start.sh && \
    echo '}' >> /start.sh && \
    echo '' >> /start.sh && \
    echo '# Start vLLM server (optional)' >> /start.sh && \
    echo 'start_vllm() {' >> /start.sh && \
    echo '    if [[ $START_VLLM == "true" ]]; then' >> /start.sh && \
    echo '        echo "Starting vLLM OpenAI API server..."' >> /start.sh && \
    echo '        cd / && \' >> /start.sh && \
    echo '        nohup python3 -m vllm.entrypoints.openai.api_server $VLLM_ARGS &> /vllm.log &' >> /start.sh && \
    echo '        echo "vLLM server started on port 8000"' >> /start.sh && \
    echo '    fi' >> /start.sh && \
    echo '}' >> /start.sh && \
    echo '' >> /start.sh && \
    echo '# ---------------------------------------------------------------------------- #' >> /start.sh && \
    echo '#                               Main Program                                   #' >> /start.sh && \
    echo '# ---------------------------------------------------------------------------- #' >> /start.sh && \
    echo '' >> /start.sh && \
    echo 'start_nginx' >> /start.sh && \
    echo 'execute_script "/pre_start.sh" "Running pre-start script..."' >> /start.sh && \
    echo 'echo "Pod Started"' >> /start.sh && \
    echo 'setup_ssh' >> /start.sh && \
    echo 'start_jupyter' >> /start.sh && \
    echo 'start_vllm' >> /start.sh && \
    echo 'export_env_vars' >> /start.sh && \
    echo 'execute_script "/post_start.sh" "Running post-start script..."' >> /start.sh && \
    echo 'echo "Start script(s) finished, pod is ready to use."' >> /start.sh && \
    echo 'sleep infinity' >> /start.sh && \
    chmod +x /start.sh
# Set permissions
RUN chmod -R 777 /workspace

# Expose ports (8888 for Jupyter, 22 for SSH, 80 for Nginx, 8000 for vLLM API)
EXPOSE 8888 22 80 8000 6006

# Health check
HEALTHCHECK --interval=30s --timeout=30s --start-period=5s --retries=3 \
    CMD curl -f http://localhost:8888/lab || exit 1

# Override the vLLM entrypoint with our start script
ENTRYPOINT ["/start.sh"]