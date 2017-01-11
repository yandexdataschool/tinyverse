#!/bin/bash
docker kill redis
docker rm redis
docker run --name redis -d --restart=always \
      --publish 0.0.0.0:7070:6379 \
        --volume /srv/docker/redis:/var/lib/redis \
          sameersbn/redis:latest --logfile /var/log/redis/redis-server.log
