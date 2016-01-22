#!/bin/bash

# Assumes all necessary packages have already been installed!!

# Get mod_wsgi
cd /tmp
sudo curl -O https://modwsgi.googlecode.com/files/mod_wsgi-3.4.tar.gz
sudo tar -xvf mod_wsgi-3.4.tar.gz
cd mod_wsgi-3.4

# Configure for Python 2.7 using python27 path
sudo ./configure --with-python=/usr/bin/python27
sudo make
sudo make install

# Once done, tell apache about new module
sudo echo 'LoadModule wsgi_module modules/mod_wsgi.so' > /etc/httpd/conf.d/wsgi.conf

