#!/bin/bash
sudo docker run -d --rm -p 6379:6379 --name redis-ai redislabs/redisai:edge-cpu-bionic