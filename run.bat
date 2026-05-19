@echo off
cd /d "C:\Users\sagar\OneDrive\Documents\TT project\TT01"

echo Compiling...
"C:\Program Files\Eclipse Adoptium\jdk-25.0.3.9-hotspot\bin\javac.exe" ^
  -cp "lib\bridj-0.7.0.jar;lib\opencv-4120.jar;lib\slf4j-api-1.7.36.jar;lib\slf4j-simple-1.7.36.jar;lib\webcam-capture-0.3.12.jar" ^
  -d bin ^
  src\App.java

if %errorlevel% neq 0 (
    echo.
    echo COMPILE FAILED. See errors above.
    pause
    exit /b 1
)

echo Compile OK. Launching...
"C:\Program Files\Eclipse Adoptium\jdk-25.0.3.9-hotspot\bin\java.exe" ^
  "-Djava.library.path=C:\Users\sagar\OneDrive\Documents\TT project\TT01" ^
  -cp "bin;lib\bridj-0.7.0.jar;lib\opencv-4120.jar;lib\slf4j-api-1.7.36.jar;lib\slf4j-simple-1.7.36.jar;lib\webcam-capture-0.3.12.jar" ^
  App
