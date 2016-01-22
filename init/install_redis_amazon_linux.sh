#!/bin/bash

# Install Linux updates, set time zones, followed by GCC and Make
sudo ln -sf /usr/share/zoneinfo/America/Los_Angeles /etc/localtime
sudo yum -y install gcc make
 
# Download, Untar and Make Redis 3.0.6
cd /tmp
wget http://download.redis.io/releases/redis-3.0.6.tar.gz
tar xzf redis-3.0.6.tar.gz
cd redis-3.0.6
make

# Create Directories and Copy Redis Files
sudo mkdir /etc/redis 
sudo mkdir /var/lib/redis
sudo cp src/redis-server src/redis-cli /usr/local/bin/
sudo cp ${PROJECTPTH}'/init/redis.conf' /etc/redis/

# Download init Script
cd /tmp
wget https://raw.github.com/saxenap/install-redis-amazon-linux-centos/master/redis-server
 
# Move and Configure Redis-Server
sudo mv redis-server /etc/init.d
sudo chmod 755 /etc/init.d/redis-server

# Auto-Enable Redis-Server
sudo chkconfig --add redis-server
sudo chkconfig --level 345 redis-server on
 
