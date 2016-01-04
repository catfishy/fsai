#!/bin/bash 

sudo yum -y update
sudo yum install ruby
sudo yum install wget
cd /home/ec2-user
wget https://aws-codedeploy-us-west-2.s3.amazonaws.com/latest/install
chmod +x ./install
sudo ./install auto
