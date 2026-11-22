@echo off
REM Installation, test de fonctionnement puis execution complete sous Windows (4 coeurs). NON TESTE sous Windows : voir GUIDE_TEST.md.
REM Prerequis : Python 3.12 (python.org, "Add to PATH") et Git for Windows. Dossier d'installation COURT recommande (ex. C:\neural_ode_symbolic).
set PYTHONUTF8=1
set PYTHONIOENCODING=utf-8
where py >nul 2>nul
if %errorlevel%==0 (set PYEXE=py -3) else (set PYEXE=python)
%PYEXE% setup_all.py
if errorlevel 1 (echo.& echo setup_all.py a signale un probleme : voir ci-dessus et setup_report.json & pause & exit /b 1)
.venv\Scripts\python.exe smoke_test.py
if errorlevel 1 (echo.& echo Le test de fonctionnement a echoue : voir smoke_report.json. Ne pas lancer l'execution complete. & pause & exit /b 1)
.venv\Scripts\python.exe checks\prereg.py verify
if errorlevel 2 .venv\Scripts\python.exe checks\prereg.py freeze
echo.
echo Test de fonctionnement reussi. Lancement de l'execution complete, PySR inclus (environ 1 h 30 a 2 h, estimation).
.venv\Scripts\python.exe run_everything.py --workers 4
pause
