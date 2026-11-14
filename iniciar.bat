@echo off
setlocal
title Monitoramento Multimodal - Tech Challenge Fase 4
cd /d "%~dp0"

echo ============================================================
echo   Monitoramento Multimodal de Pacientes - Tech Challenge F4
echo ============================================================
echo.

rem ---------- 1. Localiza um Python 3.10+ (ignora o atalho da Microsoft Store) ----------
set "PY="
call :find_python
if defined PY goto :python_ok

echo [AVISO] Nenhum Python 3.10+ foi encontrado neste computador.
echo         O "python" do Windows pode ser apenas um atalho para a Microsoft Store.
echo.
where winget >nul 2>&1
if errorlevel 1 goto :no_python
choice /c SN /m "Deseja instalar o Python 3.12 automaticamente agora"
if errorlevel 2 goto :no_python
echo.
echo Instalando o Python 3.12 - aguarde ...
winget install -e --id Python.Python.3.12 --scope user --accept-package-agreements --accept-source-agreements
call :find_python
if defined PY goto :python_ok
echo.
echo [AVISO] Instalacao concluida, mas o Python ainda nao foi localizado.
echo         Feche esta janela e execute o iniciar.bat novamente.
pause
exit /b 1

:no_python
echo [ERRO] Instale o Python 3.10+ em https://www.python.org/downloads/
echo        e marque a opcao "Add python.exe to PATH" durante a instalacao.
echo        Depois execute o iniciar.bat novamente.
pause
exit /b 1

:python_ok
echo Python encontrado:
%PY% --version
echo.

rem ---------- 2. Cria o ambiente virtual (primeira execucao) ----------
if not exist ".venv\Scripts\python.exe" (
    echo [1/3] Criando ambiente virtual .venv ...
    %PY% -m venv .venv
    if errorlevel 1 (
        echo [ERRO] Falha ao criar o ambiente virtual.
        pause
        exit /b 1
    )
)
set "VPY=.venv\Scripts\python.exe"

rem ---------- 3. Instala/atualiza dependencias quando necessario ----------
set "NEED_INSTALL=0"
if not exist ".venv\requirements.instalado" set "NEED_INSTALL=1"
if exist ".venv\requirements.instalado" (
    fc /b requirements.txt ".venv\requirements.instalado" >nul 2>&1 || set "NEED_INSTALL=1"
)
if "%NEED_INSTALL%"=="1" (
    echo [2/3] Instalando dependencias - pode levar alguns minutos na primeira vez ...
    "%VPY%" -m pip install --upgrade pip
    "%VPY%" -m pip install -r requirements.txt
    if errorlevel 1 (
        echo.
        echo [ERRO] Falha na instalacao das dependencias. Verifique sua conexao e tente novamente.
        pause
        exit /b 1
    )
    copy /y requirements.txt ".venv\requirements.instalado" >nul
) else (
    echo [2/3] Dependencias ja instaladas.
)

rem ---------- 4. Gera os dados de exemplo, se faltarem ----------
if not exist "data\samples\pose_fisio_queda.json" (
    "%VPY%" -m src.sample_data
)

rem ---------- 5. Inicia a aplicacao ----------
echo [3/3] Iniciando a aplicacao em http://localhost:8501
echo       Para encerrar, feche esta janela ou pressione Ctrl+C.
echo.
"%VPY%" -m streamlit run app.py --server.headless false --browser.gatherUsageStats false

echo.
echo Aplicacao encerrada.
pause
endlocal
exit /b 0


rem =====================================================================
rem  Sub-rotinas
rem =====================================================================

:find_python
rem Testa cada candidato executando-o de verdade: o atalho da Store falha e e ignorado.
call :try_python py -3
if defined PY exit /b 0
call :try_python python
if defined PY exit /b 0
call :try_python python3
if defined PY exit /b 0
for /d %%D in ("%LocalAppData%\Programs\Python\Python3*") do if not defined PY call :try_python "%%~D\python.exe"
if defined PY exit /b 0
for /d %%D in ("%ProgramFiles%\Python3*") do if not defined PY call :try_python "%%~D\python.exe"
if defined PY exit /b 0
for /d %%D in ("C:\Python3*") do if not defined PY call :try_python "%%~D\python.exe"
if defined PY exit /b 0
for %%P in ("%UserProfile%\anaconda3\python.exe" "%UserProfile%\miniconda3\python.exe" "%ProgramData%\anaconda3\python.exe") do if not defined PY call :try_python "%%~P"
exit /b 0

:try_python
%* -c "import sys; sys.exit(0 if sys.version_info >= (3, 10) else 1)" >nul 2>&1
if not errorlevel 1 set "PY=%*"
exit /b 0
