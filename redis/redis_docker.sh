#!/bin/bash
sudo docker run -d --rm --name=redis-dev -p=6379:6379 --hostname=redis redis:latest