# Required for our script container
FROM ubuntu:kinetic

WORKDIR /app

# Get necessary system packages
RUN apt-get update \
  && apt-get install --no-install-recommends --yes \
     build-essential \
     python3 \
     python3-pip \
     python3-dev \
  &&  pip install --upgrade pip \
  &&  apt-get install -y libmariadb-dev-compat libmariadb-dev --yes \
  &&  apt-get install mariadb-client --yes \
  && rm -rf /var/lib/apt/lists/*


# Copying all the contents from current directory in local machine to container.
COPY . .

RUN pip3 install --compile --no-cache-dir -r requirements.txt

RUN chmod +x script.sh



