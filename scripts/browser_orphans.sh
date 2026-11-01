#!/usr/bin/env bash
# Lists headless Chromium browser processes (Playwright checks) with parent, age (s) and RSS (KB). Run after every browser
# check and after stopping one: a browser whose parent is 1 or whose check has ended is a leftover. Kill only PIDs you
# inspected (kill <pid>), never by pattern: other sessions' MCP browsers run here too.
ps -eo pid,ppid,etimes,rss,args | awk 'NR == 1 || ($5 ~ /chrome-headless-shell|chromium/ && $0 !~ /--type=/) { print substr($0, 1, 170) }'
