@echo off
set CLCACHE_DIR=C:\vchigrin\browser\.clcache\
set CLCACHE_CPP2=yes
set CLCACHE_HARDLINK=yes
set CLCACHE_BASEDIR=C:\vchigrin\browser\src\
C:\Python27\python.exe "%~dp0\clcache.py" %*
