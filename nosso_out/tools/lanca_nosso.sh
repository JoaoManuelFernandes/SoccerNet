#!/bin/bash
systemctl --user is-active nosso-tudo >/dev/null 2>&1 && { echo "já está a correr"; exit 0; }
systemd-run --user --unit=nosso-tudo --collect -p MemoryMax=6900M -p MemoryHigh=6400M -p MemoryLow=5000M -p MemoryMin=4500M -p WorkingDirectory=/home/joao/soccernet/sn-gamestate /home/joao/soccernet/nosso_out/tools/corre_nosso.sh
