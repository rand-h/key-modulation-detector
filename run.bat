@echo off
setlocal enabledelayedexpansion
cd /d "%~dp0"
title ModTest - Pipeline

REM ============================================================
REM  run.bat - Pipeline de test rapide
REM ============================================================
REM  Usage :
REM    run.bat              -> test seul  (10s, defaut)
REM    run.bat verbose      -> test --verbose (detail par morceau)
REM    run.bat train        -> train + test (apres annotations/train_model)
REM    run.bat full         -> extract + train + test (nouveaux audios)
REM    run.bat extract      -> extract_test seul
REM    run.bat help         -> affiche l'aide
REM ============================================================

REM --- Detection Python ---
where python >nul 2>&1
if not errorlevel 1 (
    set "PY=python"
) else (
    where py >nul 2>&1
    if not errorlevel 1 (
        set "PY=py"
    ) else (
        echo [ERREUR] Ni 'python' ni 'py' trouve dans le PATH
        pause
        exit /b 1
    )
)

REM --- Parsing du mode ---
set "MODE=%~1"
if "%MODE%"==""          set "MODE=default"
if /I "%MODE%"=="-h"     set "MODE=help"
if /I "%MODE%"=="--help" set "MODE=help"
if /I "%MODE%"=="-v"     set "MODE=verbose"

if /I "%MODE%"=="help"    goto :show_help
if /I "%MODE%"=="default" goto :mode_default
if /I "%MODE%"=="verbose" goto :mode_verbose
if /I "%MODE%"=="train"   goto :mode_train
if /I "%MODE%"=="full"    goto :mode_full
if /I "%MODE%"=="extract" goto :mode_extract

echo [ERREUR] Mode inconnu : %MODE%
echo Utilise 'run.bat help' pour l'aide
pause
exit /b 1

REM ============================================================
REM  MODES
REM ============================================================
:mode_default
echo.
echo ============================================================
echo  MODE : test seul  (rapide)
echo ============================================================
call :check_files
call :run_test "" || goto :fail
goto :end

:mode_verbose
echo.
echo ============================================================
echo  MODE : test --verbose  (detail par morceau)
echo ============================================================
call :check_files
call :run_test "--verbose" || goto :fail
goto :end

:mode_train
echo.
echo ============================================================
echo  MODE : train + test
echo ============================================================
call :check_files
call :run_train || goto :fail
call :run_test "" || goto :fail
goto :end

:mode_full
echo.
echo ============================================================
echo  MODE : full  (extract + train + test)
echo ============================================================
call :run_extract || goto :fail
call :run_train   || goto :fail
call :run_test "" || goto :fail
goto :end

:mode_extract
echo.
echo ============================================================
echo  MODE : extract_test seul
echo ============================================================
call :run_extract || goto :fail
goto :end

REM ============================================================
REM  SOUS-ROUTINES
REM ============================================================
:check_files
if not exist "test_annotations.json" (
    echo [ATTENTION] test_annotations.json introuvable
    echo             Le test sera vide. Cree-le avant de lancer.
    echo.
)
if not exist "key_transition_model.joblib" (
    echo [INFO] key_transition_model.joblib absent
    echo        ^-^> variante 'learned' desactivee (normal au debut)
    echo        ^-^> lance 'run.bat train' pour le generer
    echo.
)
exit /b 0

:run_extract
echo.
echo ------------------------------------------------------------
echo  Extraction du test set
echo ------------------------------------------------------------
echo [start] %TIME%
%PY% extract_test.py
if errorlevel 1 (
    echo [ECHEC] extract_test.py
    exit /b 1
)
echo [ok] termine a %TIME%
exit /b 0

:run_train
echo.
echo ------------------------------------------------------------
echo  Entrainement de la matrice de transition
echo ------------------------------------------------------------
echo [start] %TIME%
%PY% train_model.py
if errorlevel 1 (
    echo [ECHEC] train_model.py
    exit /b 1
)
echo [ok] termine a %TIME%
exit /b 0

:run_test
set "EXTRA=%~1"
echo.
echo ------------------------------------------------------------
echo  Evaluation comparative
echo ------------------------------------------------------------
echo [start] %TIME%
if "!EXTRA!"=="" (
    %PY% test_model.py
) else (
    %PY% test_model.py !EXTRA!
)
if errorlevel 1 (
    echo [ECHEC] test_model.py
    exit /b 1
)
echo [ok] termine a %TIME%
exit /b 0

REM ============================================================
REM  FIN
REM ============================================================
:fail
echo.
echo ============================================================
echo  ECHEC - voir les erreurs ci-dessus
echo ============================================================
pause
exit /b 1

:end
echo.
echo ============================================================
echo  TERMINE
echo ============================================================
echo.
echo  Rappel : si tu as modifie extract_all.py, les changements
echo  sont DEJA pris en compte (imports Python frais).
echo  'run.bat full' n'est necessaire que si tu ajoutes des audios.
echo.
pause
exit /b 0

:show_help
echo.
echo ============================================================
echo  Pipeline de test - Aide
echo ============================================================
echo.
echo Usage : run.bat [mode]
echo.
echo Modes :
echo.
echo   (defaut)   test seul          ^-^> iteration rapide (10s)
echo   verbose    test --verbose     ^-^> detail par morceau
echo   train      train + test       ^-^> apres annotations / train_model
echo   full       extract + train + test  ^-^> nouveaux audios dans test/
echo   extract    extract_test seul
echo   help       affiche ce message
echo.
echo Exemples :
echo   run.bat              ^-^> test rapide
echo   run.bat verbose      ^-^> test detaille
echo   run.bat train        ^-^> re-entraine puis teste
echo   run.bat full         ^-^> apres ajout d'audios de test
echo.
pause
exit /b 0