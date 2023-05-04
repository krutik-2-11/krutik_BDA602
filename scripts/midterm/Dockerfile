# Required to have mariadb server
FROM mariadb:latest

WORKDIR /app

#Required to have mariadb client
RUN apt-get install mariadb-client --yes

COPY script.sh .

COPY  *.sql .

RUN chmod +x script.sh

