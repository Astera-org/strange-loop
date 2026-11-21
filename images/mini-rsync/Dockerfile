FROM alpine:latest

RUN apk add --no-cache rsync socat

COPY rsyncd.conf /etc/rsyncd.conf

RUN mkdir -p /data && chown nobody:nogroup /data

EXPOSE 873



