@echo off
rem MayFly as a UCI engine, for lichess-bot or any GUI that wants an executable.
rem
rem The bridge runs an executable, not a script, which is the only reason this file exists.
rem Nothing here may print to stdout, because on this channel stdout is the protocol.
rem
rem This file must have CRLF line endings. With Unix endings cmd.exe misparses it, echo stays on,
rem python never starts, and cmd reads the UCI commands off stdin as shell commands.
rem
rem VERSION picks which frozen version answers, and each plays the search it was sealed with.
rem VISITS is only used when the caller gives no clock and no node count. Lichess always sends
rem a clock, so in a real game the allocator decides and this is a fallback.
rem PYTHON is the interpreter, an environment variable (default python on PATH).

set "VERSION=mayfly_v3.pt"
set "VISITS=400"
set "LOG="
if "%PYTHON%"=="" set "PYTHON=python"

"%PYTHON%" "%~dp0uci.py"
